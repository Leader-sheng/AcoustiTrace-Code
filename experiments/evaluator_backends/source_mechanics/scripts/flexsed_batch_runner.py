from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np

from common import ensure_dir, load_config, read_csv_dicts, write_csv_dicts, safe_float


def segment_from_scores(scores: np.ndarray, step_sec: float, threshold: float) -> list[dict]:
    active = scores > threshold
    rows = []
    start = None
    peak_idx = None
    peak_val = -1.0
    for i, flag in enumerate(active):
        if flag and start is None:
            start = i
            peak_idx = i
            peak_val = float(scores[i])
        elif flag:
            if scores[i] > peak_val:
                peak_val = float(scores[i])
                peak_idx = i
        elif start is not None:
            end = i
            if end - start >= max(1, int(0.03 / max(step_sec, 1e-8))):
                rows.append((start, end, peak_idx, peak_val))
            start = None
            peak_idx = None
            peak_val = -1.0
    if start is not None:
        end = len(scores)
        if end - start >= max(1, int(0.03 / max(step_sec, 1e-8))):
            rows.append((start, end, peak_idx, peak_val))
    out = []
    for eid, (s, e, p, v) in enumerate(rows):
        out.append(
            {
                "audio_event_id": f"a{eid:04d}",
                "start_sec": s * step_sec,
                "end_sec": e * step_sec,
                "peak_sec": p * step_sec if p is not None else s * step_sec,
                "confidence": float(v),
                "raw_score": float(v),
            }
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    args = ap.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    cfg = load_config(args.config)
    out_dir = ensure_dir(Path(cfg["flexsed"]["output_dir"]))
    raw_root = ensure_dir(out_dir / "raw")
    index_rows = read_csv_dicts(Path(cfg["output"]["root"]) / "index" / "videos.csv")
    if args.shard_count < 1:
        raise SystemExit("--shard-count must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.shard_count:
        raise SystemExit("--shard-index must satisfy 0 <= shard_index < shard_count")
    if args.shard_count > 1:
        index_rows = [row for i, row in enumerate(index_rows) if i % args.shard_count == args.shard_index]
    if args.limit > 0:
        index_rows = index_rows[: args.limit]

    events = ["collision", "impact", "knock", "tap", "hit", "clap", "object falling", "object hitting floor", "bounce", "hammer hitting", "wood knocking", "metal tapping", "glass tapping", "plastic tapping", "ceramic tapping", "drop", "strike", "scrape", "clatter", "thump"]

    repo_root = Path(cfg["flexsed"]["repo_root"])
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    import api as flexsed_api
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_path = Path(
        cfg.get("flexsed", {}).get(
            "checkpoint_path", repo_root / "ckpts" / "flexsed_as.pt"
        )
    )
    clap_model_path = Path(
        cfg.get("flexsed", {}).get(
            "clap_model_path", checkpoint_path.parent / "laion-clap-htsat-unfused"
        )
    )
    if not clap_model_path.is_dir():
        raise FileNotFoundError(
            "missing FlexSED CLAP text encoder directory: "
            f"{clap_model_path}. Download laion/clap-htsat-unfused there."
        )

    # FlexSED's public API hard-codes the Hub identifier for CLAP. Redirect
    # those two constructor calls to the documented local checkpoint so a
    # release evaluation is deterministic and works without network access.
    original_clap_loader = flexsed_api.ClapTextModelWithProjection.from_pretrained
    original_tokenizer_loader = flexsed_api.AutoTokenizer.from_pretrained

    def load_local_clap(identifier, *loader_args, **loader_kwargs):
        target = clap_model_path if identifier == "laion/clap-htsat-unfused" else identifier
        if target == clap_model_path:
            loader_kwargs["local_files_only"] = True
        return original_clap_loader(str(target), *loader_args, **loader_kwargs)

    def load_local_tokenizer(identifier, *loader_args, **loader_kwargs):
        target = clap_model_path if identifier == "laion/clap-htsat-unfused" else identifier
        if target == clap_model_path:
            loader_kwargs["local_files_only"] = True
        return original_tokenizer_loader(str(target), *loader_args, **loader_kwargs)

    flexsed_api.ClapTextModelWithProjection.from_pretrained = staticmethod(load_local_clap)
    flexsed_api.AutoTokenizer.from_pretrained = staticmethod(load_local_tokenizer)

    model = flexsed_api.FlexSED(
        config_path=str(repo_root / "src" / "configs" / "model.yml"),
        ckpt_path=str(checkpoint_path),
        device=device,
    )

    summary = []
    all_rows = []
    stats = []

    for row in index_rows:
        vid = row["video_id"]
        video_path = Path(row["video_path"])
        audio_path = Path(row.get("audio_path") or "")
        if not audio_path.exists():
            audio_path = video_path.with_suffix(".wav")
        sample_dir = ensure_dir(raw_root / vid)
        result_csv = sample_dir / "events.csv"
        if args.skip_existing and result_csv.exists():
            summary.append({"video_id": vid, "status": "skipped"})
            continue

        try:
            precision = str(cfg.get("flexsed", {}).get("precision", "fp16")).lower()
            amp_dtype = None
            if device == "cuda" and precision in {"fp16", "float16", "half"}:
                amp_dtype = torch.float16
            elif device == "cuda" and precision in {"bf16", "bfloat16"}:
                amp_dtype = torch.bfloat16
            with torch.inference_mode():
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                    preds = model.run_inference(str(audio_path), events)
            preds_np = preds.squeeze(1).detach().cpu().numpy()  # [num_events, T]
            combined = preds_np.max(axis=0)
            try:
                import librosa
                duration_sec = float(librosa.get_duration(path=str(audio_path)))
            except Exception:
                duration_sec = max(1.0, len(combined) / 10.0)
            threshold = max(float(np.median(combined)) + 0.06, 0.12)
            step_sec = max(duration_sec / max(len(combined), 1), 1e-6)
            dets = segment_from_scores(combined, step_sec=step_sec, threshold=threshold)
            rows_out = []
            for i, det in enumerate(dets):
                rows_out.append(
                    {
                        "video_id": vid,
                        "video_path": str(video_path),
                        "audio_event_id": det["audio_event_id"],
                        "audio_label": "impact",
                        "start_sec": det["start_sec"],
                        "end_sec": det["end_sec"],
                        "peak_sec": det["peak_sec"],
                        "confidence": det["confidence"],
                        "raw_score": det["raw_score"],
                    }
                )
            write_csv_dicts(
                result_csv,
                rows_out,
                ["video_id", "video_path", "audio_event_id", "audio_label", "start_sec", "end_sec", "peak_sec", "confidence", "raw_score"],
            )
            summary.append({"video_id": vid, "status": "ok", "num_events": len(rows_out), "events_csv": str(result_csv)})
            all_rows.extend(rows_out)
            stats.append({"video_id": vid, "num_events": len(rows_out)})
            del preds, preds_np, combined
        except Exception as e:
            err = " ".join(str(e).split())
            lowered = err.lower()
            error_type = "flexsed_cuda_oom" if "cuda" in lowered and "out of memory" in lowered else "flexsed_inference_failed"
            summary.append({"video_id": vid, "status": "failed", "error_type": error_type, "error": err[:1000]})
        finally:
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

    write_csv_dicts(
        out_dir / "audio_events.csv",
        all_rows,
        ["video_id", "video_path", "audio_event_id", "audio_label", "start_sec", "end_sec", "peak_sec", "confidence", "raw_score"],
    )
    write_csv_dicts(out_dir / "audio_event_stats.csv", stats, ["video_id", "num_events"])

    with open(raw_root / "flexsed_raw_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
