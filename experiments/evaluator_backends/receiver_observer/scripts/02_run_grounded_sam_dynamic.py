from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import traceback
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from common import clean_text, ensure_dir, extract_frame, load_yaml, probe_video, read_csv_dicts, run_cmd, write_csv_dicts
from grounded_sam_runtime import PersistentGroundedSAM


DEFAULT_CONFIG = SCRIPT_DIR.parent / "configs" / "receiver_observer_eval_config.yaml"
DEFAULT_TARGETS = SCRIPT_DIR.parent / "configs" / "detection_targets.yaml"


_PERSISTENT_GSAM: PersistentGroundedSAM | None = None


def choose_frame_time(row: dict) -> float:
    for key in ("visual_peak_sec", "audio_peak_sec", "review_start_sec", "review_end_sec"):
        try:
            v = float(row.get(key))
            if math.isfinite(v):
                return max(0.0, v)
        except Exception:
            pass
    try:
        return max(0.0, 0.5 * (float(row.get("review_start_sec", 0.0)) + float(row.get("review_end_sec", 0.0))))
    except Exception:
        return 0.0


def run_grounded_sam(frame_path: Path, prompt: str, out_dir: Path, cfg: dict) -> tuple[bool, str]:
    global _PERSISTENT_GSAM
    frame_path = frame_path.resolve()
    out_dir = out_dir.resolve()
    ensure_dir(out_dir)
    persistent = str(cfg.get("gsam", {}).get("persistent_runtime", "true")).strip().lower() in {
        "1", "true", "yes", "y", "on",
    }
    if persistent:
        try:
            if _PERSISTENT_GSAM is None:
                _PERSISTENT_GSAM = PersistentGroundedSAM(cfg)
            _PERSISTENT_GSAM.infer(frame_path, prompt, out_dir)
            return True, ""
        except Exception:
            return False, traceback.format_exc()[-12000:]

    # Compatibility path for diagnosing third-party installations.  Production
    # evaluation uses the persistent runtime above; this deliberately retains
    # the upstream one-image demo contract when explicitly requested.
    repo = Path(cfg["runtime"]["grounded_sam_repo"])
    cmd = [
        sys.executable,
        str(repo / "grounded_sam_demo.py"),
        "--config", str(cfg["gsam"]["config"]),
        "--grounded_checkpoint", str(cfg["gsam"]["grounded_checkpoint"]),
        "--sam_version", str(cfg["gsam"]["sam_version"]),
        "--sam_checkpoint", str(cfg["gsam"]["sam_checkpoint"]),
        "--bert_base_uncased_path", str(cfg["runtime"]["bert_base_uncased_path"]),
        "--input_image", str(frame_path),
        "--text_prompt", prompt,
        "--output_dir", str(out_dir),
        "--box_threshold", str(cfg["gsam"]["box_threshold"]),
        "--text_threshold", str(cfg["gsam"]["text_threshold"]),
        "--device", str(cfg["gsam"]["device"]),
    ]
    cp = run_cmd(cmd, cwd=repo, check=False, capture_output=True)
    if cp.returncode != 0:
        # Tracebacks are emitted at the end of stderr.  Keeping the prefix used
        # to discard the actionable exception whenever warnings exceeded the
        # old 2,000-character cap.
        message = cp.stderr or cp.stdout or "gsam_failed"
        return False, message[-12000:]
    return True, ""


def safe_float(value: object, default: float = float("nan")) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def normalize_box(box: object) -> dict:
    if isinstance(box, list) and len(box) >= 4:
        return {"x1": safe_float(box[0]), "y1": safe_float(box[1]), "x2": safe_float(box[2]), "y2": safe_float(box[3])}
    if isinstance(box, dict):
        x1 = box.get("x1") if "x1" in box else box.get("x")
        y1 = box.get("y1") if "y1" in box else box.get("y")
        x2 = box.get("x2")
        y2 = box.get("y2")
        if x2 is None and "w" in box:
            x2 = safe_float(x1, 0.0) + safe_float(box.get("w"), 0.0)
        if y2 is None and "h" in box:
            y2 = safe_float(y1, 0.0) + safe_float(box.get("h"), 0.0)
        return {"x1": safe_float(x1), "y1": safe_float(y1), "x2": safe_float(x2), "y2": safe_float(y2)}
    return {}


def valid_box(box: dict) -> bool:
    vals = [safe_float(box.get(key)) for key in ("x1", "y1", "x2", "y2")]
    return all(math.isfinite(v) for v in vals) and vals[2] > vals[0] and vals[3] > vals[1]


def parse_gsam_candidates(mask_json: Path) -> list[dict]:
    if not mask_json.exists():
        return []
    try:
        obj = json.loads(mask_json.read_text(encoding="utf-8"))
    except Exception:
        return []
    raw_candidates = obj if isinstance(obj, list) else [obj]
    candidates: list[dict] = []
    for cand in raw_candidates:
        if not isinstance(cand, dict) or cand.get("label") == "background":
            continue
        box = normalize_box(cand.get("box", {}))
        if not valid_box(box):
            continue
        candidates.append(
            {
                "label": str(cand.get("label", "")),
                "confidence": safe_float(cand.get("logit", cand.get("confidence", "")), 0.0),
                "box": box,
            }
        )
    candidates.sort(key=lambda x: safe_float(x.get("confidence"), 0.0), reverse=True)
    return candidates


def dynamic_keyframe_times(video_path: Path, cfg: dict, sample_sec_override: float = 0.0, max_keyframes_override: int = 0) -> list[float]:
    info = probe_video(video_path)
    duration = safe_float(info.get("duration_sec"), 0.0)
    if duration <= 0:
        return [0.0]
    gsam_cfg = cfg.get("gsam", {})
    sample_sec = float(sample_sec_override or gsam_cfg.get("dynamic_keyframe_sample_sec", gsam_cfg.get("frame_sample_sec", 0.5)) or 0.5)
    if not math.isfinite(sample_sec) or sample_sec <= 0:
        sample_sec = 0.5
    max_keyframes = int(max_keyframes_override or gsam_cfg.get("dynamic_max_keyframes", 0) or 0)
    times = []
    t = 0.0
    while t < duration:
        times.append(round(t, 3))
        t += sample_sec
    last = max(0.0, min(duration - 1e-3, duration * 0.98))
    if not times or abs(times[-1] - last) > 0.25 * sample_sec:
        times.append(round(last, 3))
    if max_keyframes > 0 and len(times) > max_keyframes:
        if max_keyframes == 1:
            return [round(min(duration * 0.5, last), 3)]
        step = (len(times) - 1) / float(max_keyframes - 1)
        indices = sorted({round(i * step) for i in range(max_keyframes)})
        times = [times[int(i)] for i in indices]
    return times


def best_candidate(candidates: list[dict]) -> dict:
    return candidates[0] if candidates else {}


def box_center_area(box: dict, image_w: float, image_h: float) -> tuple[float, float, float]:
    w = max(1.0, float(image_w))
    h = max(1.0, float(image_h))
    x1, y1, x2, y2 = [safe_float(box.get(k), 0.0) for k in ("x1", "y1", "x2", "y2")]
    cx = 0.5 * (x1 + x2) / w
    cy = 0.5 * (y1 + y2) / h
    area = max(1.0, (x2 - x1) * (y2 - y1)) / max(1.0, w * h)
    return cx, cy, area


def box_edge_proximity_penalty(box: dict, image_w: float, image_h: float, margin_norm: float) -> float:
    """Return a soft penalty for detections touching image boundaries.

    Dynamic GSAM often returns thin false positives on frame borders. Those can
    form very stable tracks, so they need to lose against high-confidence
    in-frame detections when both pass the hard continuity gates.
    """

    if margin_norm <= 0:
        return 0.0
    w = max(1.0, float(image_w))
    h = max(1.0, float(image_h))
    x1, y1, x2, y2 = [safe_float(box.get(k), 0.0) for k in ("x1", "y1", "x2", "y2")]
    margin_x = max(1.0, margin_norm * w)
    margin_y = max(1.0, margin_norm * h)
    distances = (x1, y1, w - x2, h - y2)
    margins = (margin_x, margin_y, margin_x, margin_y)
    penalty = 0.0
    for dist, margin in zip(distances, margins):
        if dist < margin:
            penalty += min(2.0, (margin - dist) / margin)
    return penalty


def infer_image_size(rows: list[dict], fallback_w: float, fallback_h: float) -> tuple[float, float]:
    image_w = safe_float(fallback_w, 0.0)
    image_h = safe_float(fallback_h, 0.0)
    if image_w > 0 and image_h > 0:
        return image_w, image_h
    max_x = 0.0
    max_y = 0.0
    for row in rows:
        try:
            candidates = json.loads(str(row.get("candidates_json", "") or "[]"))
        except Exception:
            candidates = []
        for cand in candidates:
            box = cand.get("box", {}) if isinstance(cand, dict) else {}
            max_x = max(max_x, safe_float(box.get("x2"), 0.0))
            max_y = max(max_y, safe_float(box.get("y2"), 0.0))
    return max(1.0, image_w or max_x), max(1.0, image_h or max_y)


def associate_dynamic_candidates(rows: list[dict], cfg: dict, image_w: float, image_h: float) -> tuple[dict, int]:
    """Select one temporally coherent Grounded-SAM candidate path.

    GroundingDINO may return a high-confidence object that is not the moving
    sound source on isolated keyframes. The evaluator needs a subject track, so
    dynamic mode keeps all candidates for audit but only marks detections on the
    best continuous path as usable.
    """

    gsam_cfg = cfg.get("gsam", {})
    enabled = str(gsam_cfg.get("dynamic_track_association", "true")).strip().lower() in {"1", "true", "yes", "y", "on"}
    image_w, image_h = infer_image_size(rows, image_w, image_h)
    for row in rows:
        row["label"] = ""
        row["confidence"] = ""
        row["box_json"] = ""

    if not enabled:
        selected = {}
        selected_count = 0
        for row in rows:
            if str(row.get("status", "")) == "failed":
                continue
            try:
                candidates = json.loads(str(row.get("candidates_json", "") or "[]"))
            except Exception:
                candidates = []
            cand = best_candidate(candidates)
            if not cand:
                row["status"] = "no_detection"
                continue
            row["status"] = "success"
            row["label"] = cand.get("label", "")
            row["confidence"] = f"{safe_float(cand.get('confidence'), 0.0):.6f}"
            row["box_json"] = json.dumps(cand.get("box", {}), ensure_ascii=False)
            selected_count += 1
            if not selected or safe_float(cand.get("confidence"), 0.0) > safe_float(selected.get("confidence"), -1.0):
                selected = {
                    "label": cand.get("label", ""),
                    "confidence": cand.get("confidence", ""),
                    "box": cand.get("box", {}),
                    "frame_path": row.get("frame_path", ""),
                    "time_sec": safe_float(row.get("time_sec"), 0.0),
                    "frame_index": int(safe_float(row.get("frame_index"), 0.0)),
                }
        return selected, selected_count

    sample_sec = safe_float(gsam_cfg.get("dynamic_keyframe_sample_sec", gsam_cfg.get("frame_sample_sec", 0.5)), 0.5)
    if sample_sec <= 0:
        sample_sec = 0.5
    min_keyframes = int(gsam_cfg.get("dynamic_track_min_keyframes", 3) or 3)
    max_gap_sec = safe_float(gsam_cfg.get("dynamic_track_max_gap_sec"), 1.0)
    max_center_jump = safe_float(gsam_cfg.get("dynamic_track_max_center_jump_norm"), 0.35)
    max_log_area_jump = safe_float(gsam_cfg.get("dynamic_track_max_log_area_jump"), 2.2)
    node_base_score = safe_float(gsam_cfg.get("dynamic_track_node_base_score"), 0.05)
    conf_weight = safe_float(gsam_cfg.get("dynamic_track_confidence_weight"), 1.0)
    edge_margin_norm = safe_float(gsam_cfg.get("dynamic_track_edge_margin_norm"), 0.02)
    edge_penalty_weight = safe_float(gsam_cfg.get("dynamic_track_edge_penalty"), 1.0)
    center_penalty = safe_float(gsam_cfg.get("dynamic_track_center_jump_penalty"), 0.25)
    area_penalty = safe_float(gsam_cfg.get("dynamic_track_area_jump_penalty"), 0.05)

    nodes: list[dict] = []
    for row_idx, row in enumerate(rows):
        if str(row.get("status", "")) == "failed":
            continue
        try:
            time_sec = float(row.get("time_sec"))
        except Exception:
            continue
        try:
            candidates = json.loads(str(row.get("candidates_json", "") or "[]"))
        except Exception:
            candidates = []
        if not candidates:
            row["status"] = "no_detection"
            continue
        row["status"] = "unassociated_detection"
        row["error_message"] = "not_on_selected_dynamic_track"
        for cand_idx, cand in enumerate(candidates):
            if not isinstance(cand, dict):
                continue
            box = cand.get("box", {})
            if not valid_box(box):
                continue
            cx, cy, area = box_center_area(box, image_w, image_h)
            edge_penalty = box_edge_proximity_penalty(box, image_w, image_h, edge_margin_norm)
            confidence = safe_float(cand.get("confidence"), 0.0)
            node_score = node_base_score + conf_weight * confidence - edge_penalty_weight * edge_penalty
            nodes.append(
                {
                    "row_idx": row_idx,
                    "cand_idx": cand_idx,
                    "time_sec": time_sec,
                    "candidate": cand,
                    "cx": cx,
                    "cy": cy,
                    "area": area,
                    "confidence": confidence,
                    "node_score": node_score,
                }
            )

    if not nodes:
        return {}, 0
    nodes.sort(key=lambda x: (x["time_sec"], x["row_idx"], x["cand_idx"]))
    scores = [n["node_score"] for n in nodes]
    lengths = [1 for _ in nodes]
    prev = [-1 for _ in nodes]

    for i, cur in enumerate(nodes):
        for j in range(i):
            old = nodes[j]
            dt = cur["time_sec"] - old["time_sec"]
            if dt <= 1e-6 or dt > max_gap_sec:
                continue
            dt_factor = max(1.0, dt / sample_sec)
            center_dist = math.hypot(cur["cx"] - old["cx"], cur["cy"] - old["cy"])
            if center_dist > max_center_jump * dt_factor:
                continue
            area_ratio = max(cur["area"], 1e-9) / max(old["area"], 1e-9)
            log_area_jump = abs(math.log(area_ratio))
            if log_area_jump > max_log_area_jump * dt_factor:
                continue
            transition_cost = center_penalty * center_dist + area_penalty * log_area_jump
            candidate_score = scores[j] + cur["node_score"] - transition_cost
            candidate_length = lengths[j] + 1
            if candidate_score > scores[i] or (math.isclose(candidate_score, scores[i]) and candidate_length > lengths[i]):
                scores[i] = candidate_score
                lengths[i] = candidate_length
                prev[i] = j

    valid_terminal_indices = [idx for idx in range(len(nodes)) if lengths[idx] >= min_keyframes]
    if not valid_terminal_indices:
        return {}, 0
    best_i = max(valid_terminal_indices, key=lambda idx: (scores[idx], lengths[idx]))

    path_indices = []
    k = best_i
    while k >= 0:
        path_indices.append(k)
        k = prev[k]
    path_indices.reverse()

    selected_by_row = {nodes[idx]["row_idx"]: nodes[idx] for idx in path_indices}
    selected = {}
    for row_idx, node in selected_by_row.items():
        row = rows[row_idx]
        cand = node["candidate"]
        row["status"] = "success"
        row["label"] = cand.get("label", "")
        row["confidence"] = f"{safe_float(cand.get('confidence'), 0.0):.6f}"
        row["box_json"] = json.dumps(cand.get("box", {}), ensure_ascii=False)
        row["error_message"] = ""
        if not selected or safe_float(cand.get("confidence"), 0.0) > safe_float(selected.get("confidence"), -1.0):
            selected = {
                "label": cand.get("label", ""),
                "confidence": cand.get("confidence", ""),
                "box": cand.get("box", {}),
                "frame_path": row.get("frame_path", ""),
                "time_sec": safe_float(row.get("time_sec"), 0.0),
                "frame_index": int(safe_float(row.get("frame_index"), 0.0)),
            }
    return selected, len(selected_by_row)


def write_dynamic_rows(sample_out: Path, rows: list[dict]) -> None:
    jsonl_path = sample_out / "dynamic_bboxes.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_csv_dicts(
        sample_out / "dynamic_bboxes.csv",
        rows,
        [
            "sample_id",
            "frame_index",
            "time_sec",
            "status",
            "frame_path",
            "output_dir",
            "label",
            "confidence",
            "box_json",
            "candidates_json",
            "raw_candidate_count",
            "error_message",
        ],
    )


def dynamic_cache_is_usable(jsonl_path: Path) -> bool:
    """Return true only for a completed dynamic Grounded-SAM cache.

    Earlier releases treated the presence of ``dynamic_bboxes.jsonl`` as a
    successful cache.  A missing Python dependency could therefore leave a
    file containing only failed rows that was then skipped forever.
    """

    if not jsonl_path.is_file():
        return False
    rows: list[dict] = []
    try:
        with jsonl_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    except Exception:
        return False
    return bool(rows) and all(str(row.get("status", "")) != "failed" for row in rows)


def run_dynamic_keyframes(
    sample_id: str,
    video_path: Path,
    prompt: str,
    sample_out: Path,
    cfg: dict,
    sample_sec_override: float = 0.0,
    max_keyframes_override: int = 0,
) -> tuple[list[dict], dict, str]:
    frame_dir = ensure_dir(sample_out / "dynamic_frames")
    gsam_frame_root = ensure_dir(sample_out / "dynamic_gsam_frames")
    info = probe_video(video_path)
    fps = safe_float(info.get("fps"), 30.0)
    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0
    image_w = safe_float(info.get("width"), 0.0)
    image_h = safe_float(info.get("height"), 0.0)
    rows: list[dict] = []
    errors: list[str] = []
    for time_sec in dynamic_keyframe_times(video_path, cfg, sample_sec_override, max_keyframes_override):
        frame_index = int(round(float(time_sec) * fps))
        frame_path = frame_dir / f"frame_{frame_index:06d}.jpg"
        frame_out = gsam_frame_root / f"frame_{frame_index:06d}"
        row = {
            "sample_id": sample_id,
            "frame_index": frame_index,
            "time_sec": f"{float(time_sec):.3f}",
            "frame_path": str(frame_path),
            "output_dir": str(frame_out),
            "status": "",
            "label": "",
            "confidence": "",
            "box_json": "",
            "candidates_json": "",
            "raw_candidate_count": "0",
            "error_message": "",
        }
        if not extract_frame(video_path, float(time_sec), frame_path):
            row["status"] = "failed"
            row["error_message"] = "frame_extraction_failed"
            rows.append(row)
            errors.append(f"{time_sec:.3f}:frame_extraction_failed")
            continue
        ok, err = run_grounded_sam(frame_path, prompt, frame_out, cfg)
        if not ok:
            row["status"] = "failed"
            row["error_message"] = err
            rows.append(row)
            errors.append(f"{time_sec:.3f}:{err[:120]}")
            continue
        candidates = parse_gsam_candidates(frame_out / "mask.json")
        row["candidates_json"] = json.dumps(candidates, ensure_ascii=False)
        row["raw_candidate_count"] = str(len(candidates))
        if not candidates:
            row["status"] = "no_detection"
            rows.append(row)
            continue
        row["status"] = "candidate"
        rows.append(row)
    selected, _selected_count = associate_dynamic_candidates(rows, cfg, image_w, image_h)
    write_dynamic_rows(sample_out, rows)
    return rows, selected, "; ".join(errors[:5])


def generate_legacy_labelmaps(video_path: Path, prompt: str, sample_out: Path, cfg: dict, depth_root: Path | None = None, max_frames_override: int = 0) -> tuple[int, str]:
    frame_dir = ensure_dir(sample_out / "legacy_frames")
    gsam_frame_root = ensure_dir(sample_out / "legacy_gsam_frames")
    labelmap_dir = ensure_dir(sample_out / "legacy_labelmaps")

    frame_indices = []
    if depth_root is not None and depth_root.exists():
        for ply in sorted(depth_root.rglob("point*.ply")):
            stem = ply.stem.replace("point", "")
            try:
                frame_indices.append(int(stem))
            except Exception:
                pass
    if not frame_indices:
        info = probe_video(video_path)
        fps = float(info.get("fps") or 30.0)
        duration = float(info.get("duration_sec") or 0.0)
        if not math.isfinite(fps) or fps <= 0:
            fps = 30.0
        if not math.isfinite(duration) or duration <= 0:
            return 0, "legacy_frame_count_unknown"
        frame_indices = list(range(int(math.ceil(duration * fps))))

    max_frames = int(max_frames_override or cfg.get("gsam", {}).get("legacy_strict_max_frames", 0) or 0)
    if max_frames > 0:
        frame_indices = frame_indices[:max_frames]

    info = probe_video(video_path)
    fps = float(info.get("fps") or 30.0)
    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0

    ok_count = 0
    errors = []
    for idx in frame_indices:
        dst_labelmap = labelmap_dir / f"frame_{idx:06d}_labelmap.npy"
        if dst_labelmap.exists():
            ok_count += 1
            continue
        frame_path = frame_dir / f"frame_{idx:06d}.jpg"
        if not frame_path.exists() and not extract_frame(video_path, float(idx) / fps, frame_path):
            errors.append(f"{idx}:extract_frame_failed")
            continue
        frame_out = gsam_frame_root / f"frame_{idx:06d}"
        ok, err = run_grounded_sam(frame_path, prompt, frame_out, cfg)
        if not ok:
            errors.append(f"{idx}:{err[:200]}")
            continue
        labelmap = frame_out / "labelmap.npy"
        if not labelmap.exists():
            errors.append(f"{idx}:missing_labelmap")
            continue
        shutil.copy2(labelmap, dst_labelmap)
        ok_count += 1
    return ok_count, "; ".join(errors[:5])


def representative_frame_time(video_path: Path, row: dict, cfg: dict) -> float:
    event_clip = clean_text(row.get("event_clip_path"), "")
    if event_clip and Path(event_clip) == video_path:
        info = probe_video(video_path)
        duration = info.get("duration_sec")
        try:
            if duration and math.isfinite(float(duration)) and float(duration) > 0:
                return max(0.0, min(float(cfg["gsam"].get("frame_sample_sec", 0.5)), 0.5 * float(duration)))
        except Exception:
            pass
        return max(0.0, float(cfg["gsam"].get("frame_sample_sec", 0.5)))
    return choose_frame_time(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--targets", default=str(DEFAULT_TARGETS))
    ap.add_argument("--manifest", default="")
    ap.add_argument("--output-root", default="")
    ap.add_argument("--depth-root", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--dynamic-keyframes", action="store_true", help="Run Grounded-SAM on uniformly sampled video keyframes.")
    ap.add_argument("--single-frame-gsam", action="store_true", help="Force legacy single representative-frame Grounded-SAM.")
    ap.add_argument("--keyframe-sample-sec", type=float, default=0.0)
    ap.add_argument("--max-keyframes", type=int, default=0)
    ap.add_argument("--legacy-strict-labelmaps", action="store_true")
    ap.add_argument("--legacy-strict-max-frames", type=int, default=0)
    ap.add_argument("--status-tag", default="", help="Optional suffix for shard-specific status files.")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    target_cfg = load_yaml(args.targets)
    manifest = Path(args.manifest or (Path(cfg["output"]["root"]) / "manifests" / "receiver_observer_eval_manifest.csv"))
    output_root = Path(args.output_root or (Path(cfg["output"]["root"]) / cfg["gsam"]["output_subdir"]))
    depth_root = Path(args.depth_root or (Path(cfg["output"]["root"]) / cfg["vda"]["output_subdir"]))
    ensure_dir(output_root)

    rows = read_csv_dicts(manifest)
    if args.limit > 0:
        rows = rows[: args.limit]
    cfg_dynamic = str(cfg.get("gsam", {}).get("dynamic_keyframes", "false")).strip().lower() in {"1", "true", "yes", "y", "on"}
    use_dynamic_keyframes = (args.dynamic_keyframes or cfg_dynamic) and not args.single_frame_gsam

    out_rows = []
    errors = []
    for row in rows:
        sample_id = clean_text(row.get("sample_id"), "")
        if not sample_id or clean_text(row.get("status"), "") != "ok":
            out_rows.append({"sample_id": sample_id, "status": "skipped", "error_message": clean_text(row.get("skip_reason"), "manifest_skip")})
            continue

        video_path = Path(clean_text(row.get("event_clip_path"), clean_text(row.get("video_path"), "")) or clean_text(row.get("review_full_video_path"), ""))
        if not video_path.exists():
            out_rows.append({"sample_id": sample_id, "status": "failed", "error_message": f"missing_video:{video_path}"})
            errors.append({"sample_id": sample_id, "error": "missing_video"})
            continue

        target_key = clean_text(row.get("detection_targets_key"), "default")
        targets = [t for t in clean_text(row.get("candidate_detection_targets"), "").split("|") if t]
        prompt = f"Detect the following likely sound-source objects: {', '.join(targets) if targets else 'sound source'}."

        sample_out = output_root / sample_id
        frame_path = sample_out / "representative_frame.jpg"
        legacy_count = ""
        legacy_error = ""
        existing_ok = (
            dynamic_cache_is_usable(sample_out / "dynamic_bboxes.jsonl")
            if use_dynamic_keyframes
            else (sample_out / "bbox.json").is_file()
        )
        if args.skip_existing and existing_ok:
            if args.legacy_strict_labelmaps:
                count, legacy_error = generate_legacy_labelmaps(video_path, prompt, sample_out, cfg, depth_root / sample_id, args.legacy_strict_max_frames)
                legacy_count = str(count)
            out_rows.append(
                {
                    "sample_id": sample_id,
                    "status": "skipped_existing",
                    "output_dir": str(sample_out),
                    "confidence": "",
                    "label": "",
                    "target_key": target_key,
                    "dynamic_keyframe_count": "",
                    "dynamic_detection_count": "",
                    "legacy_labelmap_count": legacy_count,
                    "legacy_error": legacy_error,
                }
            )
            continue
        ensure_dir(sample_out)
        dynamic_keyframe_count = ""
        dynamic_detection_count = ""
        bbox = {}
        conf = ""
        label = ""
        frame_for_bbox = str(frame_path)
        if use_dynamic_keyframes:
            dynamic_rows, selected, dynamic_error = run_dynamic_keyframes(
                sample_id,
                video_path,
                prompt,
                sample_out,
                cfg,
                args.keyframe_sample_sec,
                args.max_keyframes,
            )
            dynamic_keyframe_count = str(len(dynamic_rows))
            dynamic_detection_count = str(sum(1 for r in dynamic_rows if r.get("status") == "success"))
            if not selected:
                out_rows.append(
                    {
                        "sample_id": sample_id,
                        "status": "failed",
                        "error_message": dynamic_error or "dynamic_gsam_no_detection",
                        "dynamic_keyframe_count": dynamic_keyframe_count,
                        "dynamic_detection_count": dynamic_detection_count,
                    }
                )
                errors.append({"sample_id": sample_id, "error": dynamic_error or "dynamic_gsam_no_detection"})
                continue
            bbox = selected.get("box", {}) or {}
            conf = str(selected.get("confidence", ""))
            label = str(selected.get("label", ""))
            frame_for_bbox = str(selected.get("frame_path", frame_path))
            try:
                if Path(frame_for_bbox).exists():
                    shutil.copy2(frame_for_bbox, frame_path)
            except Exception:
                pass
        else:
            if not extract_frame(video_path, representative_frame_time(video_path, row, cfg), frame_path):
                out_rows.append({"sample_id": sample_id, "status": "failed", "error_message": "frame_extraction_failed"})
                errors.append({"sample_id": sample_id, "error": "frame_extraction_failed"})
                continue

            ok, err = run_grounded_sam(frame_path, prompt, sample_out, cfg)
            if not ok:
                out_rows.append({"sample_id": sample_id, "status": "failed", "error_message": err})
                errors.append({"sample_id": sample_id, "error": err})
                continue

            candidates = parse_gsam_candidates(sample_out / "mask.json")
            cand = best_candidate(candidates)
            if cand:
                bbox = cand.get("box", {}) or {}
                conf = str(cand.get("confidence", ""))
                label = str(cand.get("label", ""))
        bbox_json = {
            "sample_id": sample_id,
            "video_id": clean_text(row.get("video_id"), ""),
            "chunk_id": clean_text(row.get("chunk_id"), ""),
            "detection_targets_key": target_key,
            "candidate_detection_targets": targets,
            "frame_path": frame_for_bbox,
            "label": label,
            "confidence": conf,
            "box": bbox,
            "tracking_mode": "dynamic_keyframe_gsam" if use_dynamic_keyframes else "single_frame_template",
        }
        (sample_out / "bbox.json").write_text(json.dumps(bbox_json, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.legacy_strict_labelmaps:
            count, legacy_error = generate_legacy_labelmaps(video_path, prompt, sample_out, cfg, depth_root / sample_id, args.legacy_strict_max_frames)
            legacy_count = str(count)
        out_rows.append(
            {
                "sample_id": sample_id,
                "status": "success",
                "output_dir": str(sample_out),
                "confidence": conf,
                "label": label,
                "target_key": target_key,
                "dynamic_keyframe_count": dynamic_keyframe_count,
                "dynamic_detection_count": dynamic_detection_count,
                "legacy_labelmap_count": legacy_count,
                "legacy_error": legacy_error,
            }
        )

    tag = clean_text(args.status_tag, "")
    suffix = f"_{tag}" if tag else ""
    status_csv = output_root / f"gsam_status{suffix}.csv"
    error_csv = output_root / f"gsam_errors{suffix}.csv"
    write_csv_dicts(
        status_csv,
        out_rows,
        [
            "sample_id",
            "status",
            "output_dir",
            "confidence",
            "label",
            "target_key",
            "dynamic_keyframe_count",
            "dynamic_detection_count",
            "legacy_labelmap_count",
            "legacy_error",
            "error_message",
        ],
    )
    write_csv_dicts(error_csv, errors, ["sample_id", "error"])
    print(json.dumps({"status_csv": str(status_csv), "error_csv": str(error_csv), "rows": len(out_rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
