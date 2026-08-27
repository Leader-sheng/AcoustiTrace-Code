from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from common import clean_text, ensure_dir, load_yaml, probe_video, read_csv_dicts, safe_float, write_csv_dicts


DEFAULT_CONFIG = SCRIPT_DIR.parent / "configs" / "receiver_observer_eval_config.yaml"


def load_bbox(sample_dir: Path) -> dict:
    bbox_path = sample_dir / "bbox.json"
    if bbox_path.exists():
        try:
            return json.loads(bbox_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def load_depth_grid(ply_path: Path, height: int, width: int) -> np.ndarray | None:
    try:
        import open3d as o3d

        pcd = o3d.io.read_point_cloud(str(ply_path))
        pts = np.asarray(pcd.points)
        if pts.size == 0:
            return None
        if pts.shape[0] != height * width:
            return None
        return pts[:, 2].reshape(height, width).astype(np.float32)
    except Exception:
        pass

    try:
        with ply_path.open("r", encoding="ascii", errors="ignore") as f:
            vertex_count = None
            for line in f:
                line = line.strip()
                if line.startswith("element vertex"):
                    vertex_count = int(line.split()[-1])
                if line == "end_header":
                    break
            if vertex_count != height * width:
                return None
            data = np.loadtxt(f, dtype=np.float32, usecols=(2,))
        if data.size != height * width:
            return None
        return data.reshape(height, width).astype(np.float32)
    except Exception:
        return None


def load_depth_stack_from_npz(sample_depth_dir: Path) -> np.ndarray | None:
    for npz_path in sorted(sample_depth_dir.glob("*_depths.npz")):
        try:
            data = np.load(npz_path)
            depths = np.asarray(data["depths"], dtype=np.float32)
            if depths.ndim == 3 and depths.shape[0] > 0:
                return depths
        except Exception:
            continue
    return None


def bbox_from_box(box: dict | list, w: int, h: int, margin: int = 24) -> tuple[int, int, int, int]:
    if not box:
        return 0, 0, w - 1, h - 1
    if isinstance(box, list) and len(box) >= 4:
        x1, y1, x2, y2 = box[:4]
    else:
        x1 = box.get("x1") if "x1" in box else box.get("x")
        y1 = box.get("y1") if "y1" in box else box.get("y")
        x2 = box.get("x2") if "x2" in box else None
        y2 = box.get("y2") if "y2" in box else None
        if x2 is None and "w" in box:
            x2 = float(x1) + float(box.get("w"))
        if y2 is None and "h" in box:
            y2 = float(y1) + float(box.get("h"))
    x1 = int(max(0, math.floor(safe_float(x1, 0.0) - margin)))
    y1 = int(max(0, math.floor(safe_float(y1, 0.0) - margin)))
    x2 = int(min(w - 1, math.ceil(safe_float(x2, w - 1) + margin)))
    y2 = int(min(h - 1, math.ceil(safe_float(y2, h - 1) + margin)))
    if x2 <= x1 or y2 <= y1:
        return 0, 0, w - 1, h - 1
    return x1, y1, x2, y2


def dynamic_bbox_rows(sample_dir: Path) -> list[dict]:
    jsonl_path = sample_dir / "dynamic_bboxes.jsonl"
    rows: list[dict] = []
    if jsonl_path.exists():
        with jsonl_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    return rows


def cached_track_is_usable(json_path: Path, dynamic_boxes_available: bool) -> bool:
    """Reject a static fallback track once dynamic detections are available."""

    if not json_path.is_file():
        return False
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    source = str(payload.get("track_source", ""))
    if dynamic_boxes_available:
        return source == "dynamic_keyframe_gsam"
    return source in {"single_frame_template", "dynamic_keyframe_gsam"}


def load_dynamic_bboxes(sample_dir: Path, w: int, h: int, margin: int) -> list[tuple[float, tuple[int, int, int, int]]]:
    boxes: list[tuple[float, tuple[int, int, int, int]]] = []
    for row in dynamic_bbox_rows(sample_dir):
        if str(row.get("status", "")) != "success":
            continue
        try:
            box_obj = json.loads(str(row.get("box_json", "") or "{}"))
        except Exception:
            box_obj = {}
        if not box_obj:
            continue
        try:
            time_sec = float(row.get("time_sec"))
        except Exception:
            continue
        if not math.isfinite(time_sec):
            continue
        box = bbox_from_box(box_obj, w, h, margin=margin)
        x1, y1, x2, y2 = box
        if x2 > x1 and y2 > y1:
            boxes.append((time_sec, box))
    boxes.sort(key=lambda x: x[0])
    deduped: list[tuple[float, tuple[int, int, int, int]]] = []
    for time_sec, box in boxes:
        if deduped and abs(deduped[-1][0] - time_sec) < 1e-6:
            deduped[-1] = (time_sec, box)
        else:
            deduped.append((time_sec, box))
    return deduped


def interpolate_bbox(
    boxes: list[tuple[float, tuple[int, int, int, int]]],
    time_sec: float,
    max_gap_sec: float,
    allow_extrapolation: bool = False,
) -> tuple[int, int, int, int] | None:
    if not boxes:
        return None
    if len(boxes) == 1:
        if abs(time_sec - boxes[0][0]) <= 1e-6:
            return boxes[0][1]
        return boxes[0][1] if allow_extrapolation and abs(time_sec - boxes[0][0]) <= max_gap_sec else None
    if time_sec < boxes[0][0]:
        return boxes[0][1] if allow_extrapolation and boxes[0][0] - time_sec <= max_gap_sec else None
    if abs(time_sec - boxes[0][0]) <= 1e-6:
        return boxes[0][1]
    if time_sec > boxes[-1][0]:
        return boxes[-1][1] if allow_extrapolation and time_sec - boxes[-1][0] <= max_gap_sec else None
    if abs(time_sec - boxes[-1][0]) <= 1e-6:
        return boxes[-1][1]
    for (t0, b0), (t1, b1) in zip(boxes[:-1], boxes[1:]):
        if t0 <= time_sec <= t1:
            if t1 - t0 > max_gap_sec:
                return None
            alpha = 0.0 if t1 <= t0 else (time_sec - t0) / (t1 - t0)
            vals = [int(round((1.0 - alpha) * float(a) + alpha * float(b))) for a, b in zip(b0, b1)]
            x1, y1, x2, y2 = vals
            if x2 <= x1 or y2 <= y1:
                return None
            return x1, y1, x2, y2
    return None


def moving_average(a: np.ndarray, win: int = 5) -> np.ndarray:
    if len(a) < win:
        return a.copy()
    k = np.ones(win) / win
    return np.convolve(a, k, mode="same")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--manifest", default="")
    ap.add_argument("--depth-root", default="")
    ap.add_argument("--gsam-root", default="")
    ap.add_argument("--output-root", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    manifest = Path(args.manifest or (Path(cfg["output"]["root"]) / "manifests" / "receiver_observer_eval_manifest.csv"))
    depth_root = Path(args.depth_root or (Path(cfg["output"]["root"]) / cfg["vda"]["output_subdir"]))
    gsam_root = Path(args.gsam_root or (Path(cfg["output"]["root"]) / cfg["gsam"]["output_subdir"]))
    output_root = Path(args.output_root or (Path(cfg["output"]["root"]) / cfg["tracks"]["output_subdir"]))
    ensure_dir(output_root)

    rows = read_csv_dicts(manifest)
    if args.limit > 0:
        rows = rows[: args.limit]

    status_rows = []
    errors = []
    for row in rows:
        sample_id = clean_text(row.get("sample_id"), "")
        if not sample_id or clean_text(row.get("status"), "") != "ok":
            status_rows.append({"sample_id": sample_id, "status": "skipped", "error_message": clean_text(row.get("skip_reason"), "manifest_skip")})
            continue

        sample_depth_dir = depth_root / sample_id
        sample_gsam_dir = gsam_root / sample_id
        sample_out = output_root / sample_id
        ensure_dir(sample_out)
        npz_path = sample_out / "track.npz"
        json_path = sample_out / "track.json"
        dynamic_boxes_available = any(
            str(item.get("status", "")) == "success"
            for item in dynamic_bbox_rows(sample_gsam_dir)
        )
        if (
            args.skip_existing
            and npz_path.is_file()
            and cached_track_is_usable(json_path, dynamic_boxes_available)
        ):
            status_rows.append({"sample_id": sample_id, "status": "skipped_existing", "npz_path": str(npz_path), "json_path": str(json_path), "error_message": ""})
            continue

        video_path = Path(clean_text(row.get("event_clip_path"), clean_text(row.get("video_path"), "")))
        if not video_path.exists():
            status_rows.append({"sample_id": sample_id, "status": "failed", "error_message": "missing_video"})
            errors.append({"sample_id": sample_id, "error": "missing_video"})
            continue
        bbox_obj = load_bbox(sample_gsam_dir)

        ply_files = sorted(sample_depth_dir.rglob("point*.ply"))
        depth_stack = None
        if not ply_files:
            depth_stack = load_depth_stack_from_npz(sample_depth_dir)
        if not ply_files and depth_stack is None:
            status_rows.append({"sample_id": sample_id, "status": "failed", "error_message": "missing_depth_outputs"})
            errors.append({"sample_id": sample_id, "error": "missing_depth_outputs"})
            continue

        try:
            probe = probe_video(video_path)
            h = int(probe.get("height") or 0)
            w = int(probe.get("width") or 0)
            if not h or not w:
                raise RuntimeError("video_dims_unknown")

            cap = None
            try:
                import cv2

                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    cap = None
            except Exception:
                cap = None

            first_depth = depth_stack[0] if depth_stack is not None else load_depth_grid(ply_files[0], h, w)
            if first_depth is None:
                raise RuntimeError("depth_grid_parse_failed")
            depth_h, depth_w = first_depth.shape[:2]
            dynamic_boxes = load_dynamic_bboxes(sample_gsam_dir, w, h, margin=cfg["tracks"]["template_margin_px"])
            dynamic_max_gap = safe_float(cfg["tracks"].get("dynamic_bbox_max_interp_gap_sec"), 0.75)
            dynamic_allow_extrapolation = str(cfg["tracks"].get("dynamic_bbox_allow_extrapolation", "false")).strip().lower() in {"1", "true", "yes", "y", "on"}
            use_dynamic_boxes = len(dynamic_boxes) > 0

            x1, y1, x2, y2 = bbox_from_box(bbox_obj.get("box", {}), w, h, margin=cfg["tracks"]["template_margin_px"])
            prev_bbox = (x1, y1, x2, y2)
            template = None
            if cap is not None and not use_dynamic_boxes:
                cap.set(1, 0)
                ok, frame = cap.read()
                if ok and frame is not None:
                    import cv2

                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    template = gray[y1:y2, x1:x2].copy() if y2 > y1 and x2 > x1 else None
            if template is None:
                template = None

            depth_median = []
            depth_min = []
            depth_mean = []
            depth_p25 = []
            depth_p75 = []
            bbox_center_x = []
            bbox_center_y = []
            bbox_area_ratio = []
            valid_mask_ratio = []
            depth_valid_ratio = []
            source_visible_ratio = []
            frame_times = []

            import cv2

            depth_count = int(depth_stack.shape[0]) if depth_stack is not None else len(ply_files)
            for i in range(depth_count):
                frame_time = float(i) / float(probe.get("fps") or 30.0)
                depth = depth_stack[i] if depth_stack is not None else load_depth_grid(ply_files[i], h, w)
                if depth is None:
                    continue
                if use_dynamic_boxes:
                    dyn_bbox = interpolate_bbox(dynamic_boxes, frame_time, dynamic_max_gap, dynamic_allow_extrapolation)
                    if dyn_bbox is None:
                        continue
                    prev_bbox = dyn_bbox
                elif cap is not None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                    ok, frame = cap.read()
                    if ok and frame is not None and template is not None and template.size > 0:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        x1p, y1p, x2p, y2p = prev_bbox
                        sx1 = max(0, x1p - cfg["tracks"]["search_margin_px"])
                        sy1 = max(0, y1p - cfg["tracks"]["search_margin_px"])
                        sx2 = min(w - 1, x2p + cfg["tracks"]["search_margin_px"])
                        sy2 = min(h - 1, y2p + cfg["tracks"]["search_margin_px"])
                        search = gray[sy1:sy2, sx1:sx2]
                        if search.size > template.size and search.shape[0] > template.shape[0] and search.shape[1] > template.shape[1]:
                            res = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
                            _, _, _, max_loc = cv2.minMaxLoc(res)
                            x1p = sx1 + max_loc[0]
                            y1p = sy1 + max_loc[1]
                            x2p = x1p + template.shape[1]
                            y2p = y1p + template.shape[0]
                            prev_bbox = (x1p, y1p, min(w - 1, x2p), min(h - 1, y2p))
                x1p, y1p, x2p, y2p = prev_bbox
                if (depth_h, depth_w) != (h, w):
                    x_scale = float(depth_w) / float(max(1, w))
                    y_scale = float(depth_h) / float(max(1, h))
                    x1d = int(round(x1p * x_scale))
                    x2d = int(round(x2p * x_scale))
                    y1d = int(round(y1p * y_scale))
                    y2d = int(round(y2p * y_scale))
                else:
                    x1d, y1d, x2d, y2d = x1p, y1p, x2p, y2p
                roi = depth[max(0, y1d) : min(depth_h, y2d), max(0, x1d) : min(depth_w, x2d)]
                if roi.size == 0:
                    roi = depth
                valid = np.isfinite(roi)
                if valid.sum() < 10:
                    roi = depth
                    valid = np.isfinite(roi)
                vals = roi[valid]
                if vals.size == 0:
                    continue
                frame_times.append(frame_time)
                depth_min.append(float(np.nanmin(vals)))
                depth_median.append(float(np.nanmedian(vals)))
                depth_mean.append(float(np.nanmean(vals)))
                depth_p25.append(float(np.nanpercentile(vals, 25)))
                depth_p75.append(float(np.nanpercentile(vals, 75)))
                bbox_center_x.append(float((x1p + x2p) / 2.0 / w))
                bbox_center_y.append(float((y1p + y2p) / 2.0 / h))
                bbox_area_ratio.append(float(max(1, (x2p - x1p) * (y2p - y1p)) / float(w * h)))
                valid_mask_ratio.append(float(valid.sum() / float(roi.size)))
                depth_valid_ratio.append(float(np.mean(np.isfinite(depth))))
                source_visible_ratio.append(float(valid.sum() / max(1, roi.size)))

            if cap is not None:
                cap.release()

            if len(depth_median) < 3:
                raise RuntimeError("insufficient_depth_track")

            depth_median = np.asarray(depth_median, dtype=np.float32)
            depth_min = np.asarray(depth_min, dtype=np.float32)
            depth_mean = np.asarray(depth_mean, dtype=np.float32)
            bbox_area_ratio = np.asarray(bbox_area_ratio, dtype=np.float32)
            depth_proxy = depth_median.copy()
            if len(depth_proxy) > 3:
                corr = np.corrcoef(depth_proxy[: len(bbox_area_ratio)], bbox_area_ratio[: len(depth_proxy)])[0, 1]
                if np.isfinite(corr) and corr > 0:
                    depth_proxy = -depth_proxy
            depth_proxy_smooth = moving_average(depth_proxy, win=min(7, max(3, len(depth_proxy) // 10 or 3)))

            np.savez_compressed(
                npz_path,
                frame_times_sec=np.asarray(frame_times, dtype=np.float32),
                source_depth_min=depth_min,
                source_depth_median=depth_median,
                source_depth_mean=depth_mean,
                source_depth_proxy=depth_proxy,
                source_depth_proxy_smooth=depth_proxy_smooth,
                bbox_center_x=np.asarray(bbox_center_x, dtype=np.float32),
                bbox_center_y=np.asarray(bbox_center_y, dtype=np.float32),
                bbox_area_ratio=np.asarray(bbox_area_ratio, dtype=np.float32),
                valid_mask_ratio=np.asarray(valid_mask_ratio, dtype=np.float32),
                depth_valid_ratio=np.asarray(depth_valid_ratio, dtype=np.float32),
                source_visible_ratio=np.asarray(source_visible_ratio, dtype=np.float32),
                track_source=np.asarray("dynamic_keyframe_gsam" if use_dynamic_boxes else "single_frame_template"),
            )
            json_path.write_text(
                json.dumps(
                    {
                        "sample_id": sample_id,
                        "track_source": "dynamic_keyframe_gsam" if use_dynamic_boxes else "single_frame_template",
                        "dynamic_keyframe_count": len(dynamic_boxes),
                        "frame_times_sec": frame_times,
                        "source_depth_min": depth_min.tolist(),
                        "source_depth_median": depth_median.tolist(),
                        "source_depth_mean": depth_mean.tolist(),
                        "source_depth_proxy": depth_proxy.tolist(),
                        "source_depth_proxy_smooth": depth_proxy_smooth.tolist(),
                        "bbox_center_x": bbox_center_x,
                        "bbox_center_y": bbox_center_y,
                        "bbox_area_ratio": bbox_area_ratio.tolist(),
                        "valid_mask_ratio": valid_mask_ratio,
                        "depth_valid_ratio": depth_valid_ratio,
                        "source_visible_ratio": source_visible_ratio,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            status_rows.append({"sample_id": sample_id, "status": "success", "npz_path": str(npz_path), "json_path": str(json_path)})
        except Exception as e:
            status_rows.append({"sample_id": sample_id, "status": "failed", "error_message": str(e)})
            errors.append({"sample_id": sample_id, "error": str(e)})

    write_csv_dicts(output_root / "track_status.csv", status_rows, ["sample_id", "status", "npz_path", "json_path", "error_message"])
    write_csv_dicts(output_root / "track_errors.csv", errors, ["sample_id", "error"])
    print(json.dumps({"status_csv": str(output_root / "track_status.csv"), "error_csv": str(output_root / "track_errors.csv"), "rows": len(status_rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
