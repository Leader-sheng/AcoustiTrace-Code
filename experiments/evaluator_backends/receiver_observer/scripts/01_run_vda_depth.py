from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from common import clean_text, ensure_dir, load_yaml, probe_video, read_csv_dicts, run_cmd, write_csv_dicts


DEFAULT_CONFIG = SCRIPT_DIR.parent / "configs" / "receiver_observer_eval_config.yaml"


def delete_visual_depth_intermediates(output_dir: Path) -> int:
    patterns = ["*.png", "*.ply", "*_depth_check.mp4", "*_vis.mp4", "*_src.mp4"]
    removed = 0
    for pattern in patterns:
        for path in output_dir.rglob(pattern):
            if path.is_file():
                try:
                    path.unlink()
                    removed += 1
                except OSError:
                    pass
    for path in sorted(output_dir.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if path.is_dir():
            try:
                path.rmdir()
            except OSError:
                pass
    return removed


def choose_video_path(row: dict) -> str:
    for key in ("event_clip_path", "video_path", "review_event_clip_path", "review_full_video_path"):
        val = clean_text(row.get(key), "")
        if val:
            return val
    return ""


def write_ascii_depth_ply(depth: np.ndarray, out_path: Path, focal_x: float = 470.4, focal_y: float = 470.4) -> None:
    """Write a minimal point cloud PLY without importing open3d.

    The downstream receiver tracker only reads xyz and reshapes z back to the
    video frame grid, so RGB/color fields are unnecessary here.
    """
    h, w = depth.shape
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    z = depth.astype(np.float32, copy=False)
    x = (xx - float(w) / 2.0) / float(focal_x) * z
    y = (yy - float(h) / 2.0) / float(focal_y) * z
    pts = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    ensure_dir(out_path.parent)
    with out_path.open("w", encoding="ascii") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {pts.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        np.savetxt(f, pts, fmt="%.6f %.6f %.6f")


def synthesize_ply_from_vda_npz(output_dir: Path) -> int:
    npz_files = sorted(output_dir.glob("*_depths.npz"))
    if not npz_files:
        return 0
    try:
        data = np.load(npz_files[0])
        depths = data["depths"]
    except Exception:
        return 0
    written = 0
    for i, depth in enumerate(depths):
        ply_path = output_dir / f"point{i:04d}.ply"
        if ply_path.exists():
            written += 1
            continue
        write_ascii_depth_ply(np.asarray(depth), ply_path)
        written += 1
    return written


def run_vda(
    video_path: Path,
    output_dir: Path,
    repo_root: str | Path,
    encoder: str,
    metric: bool,
    require_cuda: bool = True,
    fp32: bool = False,
    input_size: int = 518,
    max_res: int = 1280,
    max_len: int = -1,
    target_fps: int = -1,
    delete_visual_intermediates: bool = False,
) -> tuple[bool, str, Path]:
    video_path = video_path.resolve()
    output_dir = output_dir.resolve()
    repo_root = Path(repo_root).expanduser().resolve()
    ensure_dir(output_dir)
    log_path = output_dir / "vda_full.log"
    if require_cuda:
        cuda_probe = run_cmd(
            [
                sys.executable,
                "-c",
                "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 23)",
            ],
            check=False,
            capture_output=True,
        )
        if cuda_probe.returncode != 0:
            return False, "cuda_unavailable: VDA requires a CUDA-capable runtime", log_path
    run_args = [
        sys.executable,
        str(repo_root / "run.py"),
        "--input_video",
        str(video_path),
        "--output_dir",
        str(output_dir),
        "--encoder",
        str(encoder),
        "--input_size",
        str(input_size),
        "--max_res",
        str(max_res),
        "--max_len",
        str(max_len),
        "--target_fps",
        str(target_fps),
        "--save_npz",
    ]
    if metric:
        run_args.append("--metric")
    if fp32:
        run_args.append("--fp32")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    cp = run_cmd(run_args, cwd=repo_root, check=False, capture_output=True)
    full_log = (
        f"command: {' '.join(run_args)}\n"
        f"returncode: {cp.returncode}\n\n"
        f"--- stdout ---\n{cp.stdout or ''}\n\n"
        f"--- stderr ---\n{cp.stderr or ''}\n"
    )
    log_path.write_text(full_log, encoding="utf-8", errors="replace")
    if os.environ.get("SOUNDPHYSICS_VDA_SYNTH_PLY", "").strip() == "1" and len(list(output_dir.rglob("point*.ply"))) == 0:
        synthesize_ply_from_vda_npz(output_dir)
    ply_count = len(list(output_dir.rglob("point*.ply")))
    npz_count = len(list(output_dir.glob("*_depths.npz")))
    if delete_visual_intermediates:
        delete_visual_depth_intermediates(output_dir)
    if ply_count == 0 and npz_count == 0:
        if cp.returncode != 0:
            err = cp.stderr or cp.stdout or "vda_failed"
            return False, f"{err[:1800]}\nfull_log:{log_path}", log_path
        return False, f"no_depth_outputs\nfull_log:{log_path}", log_path
    return True, "", log_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--manifest", default="")
    ap.add_argument("--output-root", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--status-tag", default="", help="Optional suffix for shard-specific status files.")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    output_root = Path(args.output_root or (Path(cfg["output"]["root"]) / cfg["vda"]["output_subdir"]))
    manifest = Path(args.manifest or (Path(cfg["output"]["root"]) / "manifests" / "receiver_observer_eval_manifest.csv"))
    ensure_dir(output_root)

    rows = read_csv_dicts(manifest)
    if args.limit > 0:
        rows = rows[: args.limit]

    status_rows = []
    errors = []
    counts = Counter()
    for row in rows:
        sample_id = clean_text(row.get("sample_id"), "")
        if not sample_id:
            counts["missing_sample_id"] += 1
            continue
        if clean_text(row.get("status"), "") != "ok":
            status_rows.append(
                {
                    "sample_id": sample_id,
                    "status": "skipped",
                    "output_dir": "",
                    "error_message": clean_text(row.get("skip_reason"), "manifest_skip"),
                    "num_ply_files": 0,
                    "num_npz_files": 0,
                    "log_path": "",
                }
            )
            counts["skipped"] += 1
            continue

        video_path = Path(choose_video_path(row))
        if not video_path.exists():
            status_rows.append(
                {
                    "sample_id": sample_id,
                    "status": "failed",
                    "output_dir": "",
                    "error_message": f"missing_video:{video_path}",
                    "num_ply_files": 0,
                    "num_npz_files": 0,
                    "log_path": "",
                }
            )
            errors.append({"sample_id": sample_id, "error": "missing_video", "video_path": str(video_path)})
            counts["missing_video"] += 1
            continue

        sample_out = output_root / sample_id
        if args.skip_existing and (sample_out / "vda_status.json").exists():
            counts["skip_existing"] += 1
            status_rows.append(
                {
                    "sample_id": sample_id,
                    "status": "skipped_existing",
                    "output_dir": str(sample_out),
                    "error_message": "",
                    "num_ply_files": len(list(sample_out.rglob("point*.ply"))),
                    "num_npz_files": len(list(sample_out.glob("*_depths.npz"))),
                    "log_path": str(sample_out / "vda_full.log"),
                }
            )
            continue

        ok, err, log_path = run_vda(
            video_path,
            sample_out,
            cfg["runtime"]["vda_repo"],
            cfg["vda"]["encoder"],
            bool(cfg["vda"]["metric"]),
            bool(cfg.get("vda", {}).get("require_cuda", True)),
            bool(cfg.get("vda", {}).get("fp32", False)),
            int(cfg.get("vda", {}).get("input_size", 518)),
            int(cfg.get("vda", {}).get("max_res", 1280)),
            int(cfg.get("vda", {}).get("max_len", -1)),
            int(cfg.get("vda", {}).get("target_fps", -1)),
            bool(cfg.get("vda", {}).get("delete_visual_intermediates", False)),
        )
        ply_count = len(list(sample_out.rglob("point*.ply"))) if sample_out.exists() else 0
        npz_count = len(list(sample_out.glob("*_depths.npz"))) if sample_out.exists() else 0
        status = "success" if ok else "failed"
        status_rows.append(
            {
                "sample_id": sample_id,
                "status": status,
                "output_dir": str(sample_out),
                "error_message": err,
                "num_ply_files": ply_count,
                "num_npz_files": npz_count,
                "log_path": str(log_path),
            }
        )
        if not ok:
            errors.append({"sample_id": sample_id, "error": err, "video_path": str(video_path)})
            counts["failed"] += 1
        else:
            counts["success"] += 1
            with open(sample_out / "vda_status.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "sample_id": sample_id,
                        "video_path": str(video_path),
                        "output_dir": str(sample_out),
                        "num_ply_files": ply_count,
                        "num_npz_files": npz_count,
                        "log_path": str(log_path),
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

    tag = clean_text(args.status_tag, "")
    suffix = f"_{tag}" if tag else ""
    status_csv = output_root / f"depth_status{suffix}.csv"
    error_csv = output_root / f"depth_errors{suffix}.csv"
    write_csv_dicts(status_csv, status_rows, ["sample_id", "status", "output_dir", "error_message", "num_ply_files", "num_npz_files", "log_path"])
    write_csv_dicts(error_csv, errors, ["sample_id", "error", "video_path"])
    print(json.dumps({"counts": dict(counts), "status_csv": str(status_csv), "error_csv": str(error_csv)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
