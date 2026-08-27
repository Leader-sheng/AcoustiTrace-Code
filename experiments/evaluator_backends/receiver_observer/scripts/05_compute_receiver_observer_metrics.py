from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from common import clean_text, ensure_dir, load_yaml, read_csv_dicts, write_csv_dicts


DEFAULT_CONFIG = SCRIPT_DIR.parent / "configs" / "receiver_observer_eval_config.yaml"


def corr_safe(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    if len(a) < 3 or len(b) < 3:
        return float("nan"), float("nan")
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan"), float("nan")
    a = a[m]
    b = b[m]
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return float("nan"), float("nan")
    try:
        from scipy.stats import pearsonr, spearmanr

        return float(spearmanr(a, b).correlation), float(pearsonr(a, b).statistic)
    except Exception:
        return float(np.corrcoef(a, b)[0, 1]), float(np.corrcoef(a, b)[0, 1])


def linfit_slope(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return float("nan")
    try:
        return float(np.polyfit(x[m], y[m], 1)[0])
    except Exception:
        return float("nan")


def r2_proxy(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    try:
        p = np.polyfit(x[m], y[m], 1)
        pred = np.polyval(p, x[m])
        ss_res = float(np.sum((y[m] - pred) ** 2))
        ss_tot = float(np.sum((y[m] - np.mean(y[m])) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    except Exception:
        return float("nan")


def windowed_distance_r2_search(
    dist: np.ndarray,
    spl: np.ndarray,
    times: np.ndarray,
    window_sec: float,
    slide_sec: float,
    search_min_n: float,
    search_max_n: float,
    search_density: int,
) -> tuple[float, float, float, float]:
    m = np.isfinite(dist) & np.isfinite(spl) & np.isfinite(times)
    if m.sum() < 4:
        return float("nan"), float("nan"), float("nan"), float("nan")
    dist = np.asarray(dist[m], dtype=np.float32)
    spl = np.asarray(spl[m], dtype=np.float32)
    times = np.asarray(times[m], dtype=np.float32)
    if len(times) < 4:
        return float("nan"), float("nan"), float("nan"), float("nan")
    if not np.isfinite(window_sec) or window_sec <= 0 or not np.isfinite(slide_sec) or slide_sec <= 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    if not np.isfinite(search_min_n) or not np.isfinite(search_max_n) or search_max_n <= search_min_n:
        return float("nan"), float("nan"), float("nan"), float("nan")
    if search_density <= 0:
        return float("nan"), float("nan"), float("nan"), float("nan")

    n_grid = np.linspace(search_min_n, search_max_n, search_density)
    best_r2 = float("-inf")
    best_start = float("nan")
    best_end = float("nan")
    best_n = float("nan")

    start = float(times[0])
    last = float(times[-1])
    while start + window_sec <= last + 1e-9:
        end = start + window_sec
        w = (times >= start) & (times <= end)
        if w.sum() >= 3:
            w_dist = dist[w]
            w_spl = spl[w]
            w_spl_norm = w_spl - w_spl[0]
            d0 = float(w_dist[0])
            if np.isfinite(d0) and d0 > 0 and np.nanmax(w_dist) > 0:
                base = np.log10(np.maximum(d0 / np.maximum(w_dist, 1e-6), 1e-6))
                ss_tot = float(np.sum((w_spl_norm - np.mean(w_spl_norm)) ** 2))
                if ss_tot > 0:
                    for n_test in n_grid:
                        theory = n_test * base
                        ss_res = float(np.sum((w_spl_norm - theory) ** 2))
                        r2 = 1.0 - ss_res / ss_tot
                        if np.isfinite(r2) and r2 > best_r2:
                            best_r2 = float(r2)
                            best_start = start
                            best_end = end
                            best_n = float(n_test)
        start += slide_sec
    if best_r2 == float("-inf"):
        return float("nan"), float("nan"), float("nan"), float("nan")
    return best_r2, best_start, best_end, best_n


def legacy_spl_curve(audio_path: Path, sr: int, fps: int, frame_length: int, smooth_sigma: float) -> tuple[np.ndarray, np.ndarray]:
    import librosa
    from scipy.ndimage import gaussian_filter1d

    y, sr_loaded = librosa.load(str(audio_path), sr=sr, mono=True)
    hop_length = max(1, int(sr_loaded / fps))
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    spl = 20.0 * np.log10(gaussian_filter1d(rms, sigma=smooth_sigma) + 1e-9)
    spl -= np.nanmax(spl)
    times = librosa.frames_to_time(np.arange(len(spl)), sr=sr_loaded, hop_length=hop_length)
    return times.astype(np.float32), spl.astype(np.float32)


def legacy_strict_r2_from_files(audio_path: Path, depth_dir: Path, gsam_dir: Path, cfg: dict) -> tuple[float, float, int, float, float]:
    try:
        import open3d as o3d
    except Exception:
        return float("nan"), float("nan"), 0, float("nan"), float("nan")

    fps = int(cfg["metrics"]["legacy_audio_fps"])
    at, spl = legacy_spl_curve(
        audio_path,
        int(cfg["metrics"]["legacy_audio_sample_rate"]),
        fps,
        int(cfg["metrics"]["legacy_audio_frame_length"]),
        float(cfg["metrics"]["legacy_audio_smooth_sigma"]),
    )
    labelmap_dir = gsam_dir / "legacy_labelmaps"
    if not labelmap_dir.exists():
        return float("nan"), float("nan"), 0, float("nan"), float("nan")

    all_t = []
    all_d = []
    all_s = []
    for labelmap_path in sorted(labelmap_dir.glob("frame_*_labelmap.npy")):
        try:
            frame_idx = int(labelmap_path.stem.split("_")[1])
        except Exception:
            continue
        if frame_idx >= len(spl):
            continue
        ply_path = depth_dir / f"point{frame_idx:04d}.ply"
        if not ply_path.exists():
            continue
        mask = np.load(labelmap_path)
        pcd = o3d.io.read_point_cloud(str(ply_path))
        pts = np.asarray(pcd.points)
        if pts.size == 0 or pts.shape[0] != mask.size:
            continue
        try:
            depth_map = pts[:, 2].reshape(mask.shape)
        except Exception:
            continue
        obj_mask = mask == 1
        if np.sum(obj_mask) < 15:
            continue
        vals = depth_map[obj_mask]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        all_t.append(float(frame_idx) / float(fps))
        all_d.append(float(np.nanmin(vals)))
        all_s.append(float(spl[frame_idx]))

    if len(all_t) < 5:
        return float("nan"), float("nan"), len(all_t), float("nan"), float("nan")
    r2, start, end, best_n = windowed_distance_r2_search(
        np.asarray(all_d, dtype=np.float32),
        np.asarray(all_s, dtype=np.float32),
        np.asarray(all_t, dtype=np.float32),
        float(cfg["metrics"]["distance_r2_window_sec"]),
        float(cfg["metrics"]["distance_r2_slide_sec"]),
        float(cfg["metrics"]["distance_r2_search_min_n"]),
        float(cfg["metrics"]["distance_r2_search_max_n"]),
        int(cfg["metrics"]["distance_r2_search_density"]),
    )
    return r2, best_n, len(all_t), start, end


def lag_proxy(x: np.ndarray, y: np.ndarray, times: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(times)
    if m.sum() < 4:
        return float("nan")
    x = np.asarray(x[m], dtype=np.float32)
    y = np.asarray(y[m], dtype=np.float32)
    times = np.asarray(times[m], dtype=np.float32)
    dt = float(np.nanmedian(np.diff(times))) if len(times) > 1 else float("nan")
    if not np.isfinite(dt) or dt <= 0:
        return float("nan")
    x = x - np.mean(x)
    y = y - np.mean(y)
    if np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return float("nan")
    corr = np.correlate(y, x, mode="full")
    lag_idx = int(np.argmax(corr)) - (len(x) - 1)
    return float(lag_idx * dt)


def legacy_approaching_window(
    dist: np.ndarray,
    spl: np.ndarray,
    times: np.ndarray,
    window_sec: float,
    slide_sec: float,
    min_dist_range: float,
    min_dist_slope: float,
) -> tuple[bool, float, float, float, float, float, float]:
    m = np.isfinite(dist) & np.isfinite(spl) & np.isfinite(times)
    if m.sum() < 4:
        return False, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
    dist = np.asarray(dist[m], dtype=np.float32)
    spl = np.asarray(spl[m], dtype=np.float32)
    times = np.asarray(times[m], dtype=np.float32)
    best = None
    start = float(times[0])
    last = float(times[-1])
    while start + window_sec <= last + 1e-9:
        end = start + window_sec
        w = (times >= start) & (times <= end)
        if w.sum() >= 3:
            wd = dist[w]
            ws = spl[w]
            wt = times[w]
            d_range = float(np.nanmax(wd) - np.nanmin(wd))
            d_slope = linfit_slope(wt, wd)
            s_slope = linfit_slope(wt, ws)
            corr, _ = corr_safe(wd, ws)
            monotonic = float(np.mean(np.sign(np.diff(wd)) * np.sign(np.diff(ws)) <= 0)) if len(wd) > 2 else float("nan")
            applicable = d_range >= min_dist_range and d_slope < -min_dist_slope
            score = float(np.nanmean([
                1.0 - max(0.0, corr if np.isfinite(corr) else 0.0),
                min(1.0, max(0.0, -d_slope * 10.0)) if np.isfinite(d_slope) else 0.0,
                min(1.0, max(0.0, s_slope / 10.0)) if np.isfinite(s_slope) else 0.0,
                monotonic if np.isfinite(monotonic) else 0.0,
            ]))
            if applicable and (best is None or score > best[0]):
                best = (score, start, end, d_slope, s_slope, corr, monotonic)
        start += slide_sec
    if best is None:
        return False, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
    score, start, end, d_slope, s_slope, corr, monotonic = best
    return True, float(start), float(end), float(d_slope), float(s_slope), float(corr), float(score)


def legacy_lateral_window(
    dist: np.ndarray,
    spl: np.ndarray,
    bbox_area: np.ndarray,
    times: np.ndarray,
    window_sec: float,
    slide_sec: float,
    max_distance_cv: float,
    min_bbox_motion: float,
) -> tuple[bool, float, float, float, float, float, float, float]:
    m = np.isfinite(dist) & np.isfinite(spl) & np.isfinite(bbox_area) & np.isfinite(times)
    if m.sum() < 4:
        return False, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
    dist = np.asarray(dist[m], dtype=np.float32)
    spl = np.asarray(spl[m], dtype=np.float32)
    bbox_area = np.asarray(bbox_area[m], dtype=np.float32)
    times = np.asarray(times[m], dtype=np.float32)
    best = None
    start = float(times[0])
    last = float(times[-1])
    while start + window_sec <= last + 1e-9:
        end = start + window_sec
        w = (times >= start) & (times <= end)
        if w.sum() >= 3:
            wd = dist[w]
            ws = spl[w]
            wb = bbox_area[w]
            distance_cv = float(np.nanstd(wd) / (np.nanmean(np.abs(wd)) + 1e-8))
            bbox_motion = float(np.nanmean(np.abs(np.diff(wb)))) if len(wb) > 1 else float("nan")
            spl_variance = float(np.nanvar(ws))
            spl_drift = float(np.nanmax(ws) - np.nanmin(ws))
            applicable = distance_cv <= max_distance_cv and bbox_motion >= min_bbox_motion
            stability_score = float(1.0 / (1.0 + spl_variance + max(0.0, spl_drift) / 10.0)) if np.isfinite(spl_variance) else float("nan")
            if applicable and np.isfinite(stability_score) and (best is None or stability_score > best[0]):
                best = (stability_score, start, end, distance_cv, bbox_motion, spl_variance, spl_drift)
        start += slide_sec
    if best is None:
        return False, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
    score, start, end, distance_cv, bbox_motion, spl_variance, spl_drift = best
    return True, float(start), float(end), float(distance_cv), float(bbox_motion), float(spl_variance), float(spl_drift), float(score)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--manifest", default="")
    ap.add_argument("--audio-root", default="")
    ap.add_argument("--track-root", default="")
    ap.add_argument("--output-root", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    manifest = Path(args.manifest or (Path(cfg["output"]["root"]) / "manifests" / "receiver_observer_eval_manifest.csv"))
    audio_root = Path(args.audio_root or (Path(cfg["output"]["root"]) / cfg["audio"]["output_subdir"]))
    track_root = Path(args.track_root or (Path(cfg["output"]["root"]) / cfg["tracks"]["output_subdir"]))
    output_root = Path(args.output_root or (Path(cfg["output"]["root"]) / "metrics"))
    ensure_dir(output_root)

    rows = read_csv_dicts(manifest)
    if args.limit > 0:
        rows = rows[: args.limit]

    out_rows = []
    out_jsonl = output_root / "receiver_observer_metrics.jsonl"
    if out_jsonl.exists() and not args.skip_existing:
        out_jsonl.unlink()

    for row in rows:
        sample_id = clean_text(row.get("sample_id"), "")
        if not sample_id or clean_text(row.get("status"), "") != "ok":
            out_rows.append({"sample_id": sample_id, "status": "skipped", "skip_reason": clean_text(row.get("skip_reason"), "manifest_skip"), "applicable_testpoints": ""})
            continue

        audio_npz = audio_root / sample_id / "audio_features.npz"
        track_npz = track_root / sample_id / "track.npz"
        if not audio_npz.exists() or not track_npz.exists():
            out_rows.append({"sample_id": sample_id, "status": "failed", "skip_reason": "missing_audio_or_track", "applicable_testpoints": ""})
            continue

        try:
            a = np.load(audio_npz)
            t = np.load(track_npz)
            at = np.asarray(a["frame_times_sec"], dtype=np.float32)
            spl = np.asarray(a["spl_curve_db"], dtype=np.float32)
            loud = np.asarray(a["loudness_curve_db"], dtype=np.float32)
            tt = np.asarray(t["frame_times_sec"], dtype=np.float32)
            dist = np.asarray(t["source_depth_proxy_smooth"], dtype=np.float32)
            raw_dist = np.asarray(t["source_depth_proxy"], dtype=np.float32)
            legacy_dist = np.asarray(t["source_depth_min"] if "source_depth_min" in t else t["source_depth_median"], dtype=np.float32)
            bbox_area = np.asarray(t["bbox_area_ratio"], dtype=np.float32)

            if len(at) < 3 or len(tt) < 3:
                raise RuntimeError("insufficient_curve_points")

            common_t = np.intersect1d(np.round(at, 2), np.round(tt, 2))
            if len(common_t) < 3:
                common_t = np.linspace(max(at.min(), tt.min()), min(at.max(), tt.max()), min(len(at), len(tt), 50))
            spl_i = np.interp(common_t, at, spl)
            loud_i = np.interp(common_t, at, loud)
            dist_i = np.interp(common_t, tt, dist)
            raw_dist_i = np.interp(common_t, tt, raw_dist)
            bbox_i = np.interp(common_t, tt, bbox_area)

            valid_aligned_frame_ratio = float(np.mean(np.isfinite(spl_i) & np.isfinite(dist_i)))
            dist_range = float(np.nanmax(dist_i) - np.nanmin(dist_i))
            spl_range = float(np.nanmax(spl_i) - np.nanmin(spl_i))
            dist_decay_applicable = bool(dist_range >= cfg["metrics"]["distance_decay_min_range"] and valid_aligned_frame_ratio >= 0.5)
            spearman_corr, pearson_corr = corr_safe(dist_i, spl_i)
            log_dist = np.log(np.maximum(np.asarray(dist_i) - np.nanmin(dist_i) + 1e-6, 1e-6))
            log_distance_spl_slope = linfit_slope(log_dist, spl_i)
            inverse_square_fit_r2_proxy = r2_proxy(log_dist, spl_i)
            window_r2_proxy, window_r2_start_sec, window_r2_end_sec, window_r2_best_n = windowed_distance_r2_search(
                dist_i,
                spl_i,
                common_t,
                float(cfg["metrics"]["distance_r2_window_sec"]),
                float(cfg["metrics"]["distance_r2_slide_sec"]),
                float(cfg["metrics"]["distance_r2_search_min_n"]),
                float(cfg["metrics"]["distance_r2_search_max_n"]),
                int(cfg["metrics"]["distance_r2_search_density"]),
            )
            legacy_window_r2_proxy = float("nan")
            legacy_window_r2_start_sec = float("nan")
            legacy_window_r2_end_sec = float("nan")
            legacy_window_r2_best_n = float("nan")
            legacy_approaching_applicable = False
            legacy_approaching_start_sec = float("nan")
            legacy_approaching_end_sec = float("nan")
            legacy_approaching_distance_slope = float("nan")
            legacy_approaching_spl_slope = float("nan")
            legacy_approaching_corr = float("nan")
            legacy_approaching_score = float("nan")
            legacy_lateral_applicable = False
            legacy_lateral_start_sec = float("nan")
            legacy_lateral_end_sec = float("nan")
            legacy_lateral_distance_cv = float("nan")
            legacy_lateral_bbox_motion = float("nan")
            legacy_lateral_spl_variance_db = float("nan")
            legacy_lateral_spl_drift_db = float("nan")
            legacy_lateral_stability_score = float("nan")
            legacy_strict_r2 = float("nan")
            legacy_strict_best_n = float("nan")
            legacy_strict_frames = 0
            legacy_strict_start_sec = float("nan")
            legacy_strict_end_sec = float("nan")
            event_audio_raw = clean_text(row.get("event_audio_path"), "")
            event_audio_path = (
                Path(event_audio_raw)
                if event_audio_raw
                else audio_root / sample_id / "extracted_audio.wav"
            )
            if event_audio_path.is_file():
                legacy_at, legacy_spl = legacy_spl_curve(
                    event_audio_path,
                    int(cfg["metrics"]["legacy_audio_sample_rate"]),
                    int(cfg["metrics"]["legacy_audio_fps"]),
                    int(cfg["metrics"]["legacy_audio_frame_length"]),
                    float(cfg["metrics"]["legacy_audio_smooth_sigma"]),
                )
                legacy_t = np.intersect1d(np.round(legacy_at, 2), np.round(tt, 2))
                if len(legacy_t) < 3:
                    legacy_t = np.linspace(max(legacy_at.min(), tt.min()), min(legacy_at.max(), tt.max()), min(len(legacy_at), len(tt), 50))
                legacy_spl_i = np.interp(legacy_t, legacy_at, legacy_spl)
                legacy_dist_i = np.interp(legacy_t, tt, legacy_dist)
                legacy_bbox_i = np.interp(legacy_t, tt, bbox_area)
                legacy_window_r2_proxy, legacy_window_r2_start_sec, legacy_window_r2_end_sec, legacy_window_r2_best_n = windowed_distance_r2_search(
                    legacy_dist_i,
                    legacy_spl_i,
                    legacy_t,
                    float(cfg["metrics"]["distance_r2_window_sec"]),
                    float(cfg["metrics"]["distance_r2_slide_sec"]),
                    float(cfg["metrics"]["distance_r2_search_min_n"]),
                    float(cfg["metrics"]["distance_r2_search_max_n"]),
                    int(cfg["metrics"]["distance_r2_search_density"]),
                )
                (
                    legacy_approaching_applicable,
                    legacy_approaching_start_sec,
                    legacy_approaching_end_sec,
                    legacy_approaching_distance_slope,
                    legacy_approaching_spl_slope,
                    legacy_approaching_corr,
                    legacy_approaching_score,
                ) = legacy_approaching_window(
                    legacy_dist_i,
                    legacy_spl_i,
                    legacy_t,
                    float(cfg["metrics"]["distance_r2_window_sec"]),
                    float(cfg["metrics"]["distance_r2_slide_sec"]),
                    float(cfg["metrics"]["distance_decay_min_range"]),
                    float(cfg["metrics"]["approaching_min_slope"]),
                )
                (
                    legacy_lateral_applicable,
                    legacy_lateral_start_sec,
                    legacy_lateral_end_sec,
                    legacy_lateral_distance_cv,
                    legacy_lateral_bbox_motion,
                    legacy_lateral_spl_variance_db,
                    legacy_lateral_spl_drift_db,
                    legacy_lateral_stability_score,
                ) = legacy_lateral_window(
                    legacy_dist_i,
                    legacy_spl_i,
                    legacy_bbox_i,
                    legacy_t,
                    float(cfg["metrics"]["distance_r2_window_sec"]),
                    float(cfg["metrics"]["distance_r2_slide_sec"]),
                    float(cfg["metrics"]["lateral_max_distance_cv"]),
                    float(cfg["metrics"]["lateral_min_bbox_motion"]),
                )
                legacy_strict_r2, legacy_strict_best_n, legacy_strict_frames, legacy_strict_start_sec, legacy_strict_end_sec = legacy_strict_r2_from_files(
                    event_audio_path,
                    track_root.parent / cfg["vda"]["output_subdir"] / sample_id,
                    track_root.parent / cfg["gsam"]["output_subdir"] / sample_id,
                    cfg,
                )
            distance_spl_lag_sec = lag_proxy(dist_i, spl_i, common_t)

            d_slope = linfit_slope(common_t, dist_i)
            s_slope = linfit_slope(common_t, spl_i)
            approaching_applicable = bool(d_slope < -cfg["metrics"]["approaching_min_slope"] and dist_range >= cfg["metrics"]["distance_decay_min_range"])
            approaching_segment_start_sec = float(common_t[0]) if approaching_applicable else float("nan")
            approaching_segment_end_sec = float(common_t[-1]) if approaching_applicable else float("nan")
            approaching_corr, _ = corr_safe(dist_i, loud_i)
            monotonic_agreement_ratio = float(np.mean(np.sign(np.diff(dist_i)) * np.sign(np.diff(loud_i)) <= 0)) if len(dist_i) > 2 else float("nan")
            approaching_consistency_score = float(np.nanmean([1.0 - max(0.0, approaching_corr or 0.0), min(1.0, max(0.0, -d_slope * 10.0)), monotonic_agreement_ratio if np.isfinite(monotonic_agreement_ratio) else 0.0]))

            distance_cv = float(np.nanstd(dist_i) / (np.nanmean(np.abs(dist_i)) + 1e-8))
            bbox_motion_magnitude = float(np.nanmean(np.abs(np.diff(bbox_i)))) if len(bbox_i) > 1 else float("nan")
            lateral_applicable = bool(distance_cv <= cfg["metrics"]["lateral_max_distance_cv"] and bbox_motion_magnitude >= cfg["metrics"]["lateral_min_bbox_motion"])
            lateral_segment_start_sec = float(common_t[0]) if lateral_applicable else float("nan")
            lateral_segment_end_sec = float(common_t[-1]) if lateral_applicable else float("nan")
            spl_variance_db = float(np.nanvar(spl_i))
            spl_drift_db = float(np.nanmax(spl_i) - np.nanmin(spl_i))
            spl_peak_to_trough_db = spl_drift_db
            loudness_stability_score = float(1.0 / (1.0 + spl_variance_db)) if np.isfinite(spl_variance_db) else float("nan")
            lateral_distance_stability_score = float(1.0 / (1.0 + distance_cv)) if np.isfinite(distance_cv) else float("nan")

            row_out = {
                "sample_id": sample_id,
                "video_id": clean_text(row.get("video_id"), ""),
                "chunk_id": clean_text(row.get("chunk_id"), ""),
                "event_clip_path": clean_text(row.get("event_clip_path"), ""),
                "event_audio_path": clean_text(row.get("event_audio_path"), ""),
                "detection_targets": clean_text(row.get("candidate_detection_targets"), ""),
                "source_selection_confidence": clean_text(row.get("confidence"), ""),
                "valid_mask_ratio": f"{float(np.nanmean(np.asarray(t['valid_mask_ratio'], dtype=np.float32))):.6f}" if "valid_mask_ratio" in t else "",
                "source_visible_ratio": f"{float(np.nanmean(np.asarray(t['source_visible_ratio'], dtype=np.float32))):.6f}" if "source_visible_ratio" in t else "",
                "depth_valid_ratio": f"{float(np.nanmean(np.asarray(t['depth_valid_ratio'], dtype=np.float32))):.6f}" if "depth_valid_ratio" in t else "",
                "audio_valid_ratio": f"{float(np.mean(np.isfinite(spl_i))):.6f}",
                "evaluation_status": "success",
                "skip_reason": "",
                "applicable_testpoints": "|".join([tp for tp, ok in [("distance_decay", dist_decay_applicable), ("approaching", approaching_applicable), ("lateral", lateral_applicable)] if ok]),
                "distance_decay_applicable": dist_decay_applicable,
                "approaching_applicable": approaching_applicable,
                "lateral_applicable": lateral_applicable,
                "distance_spl_spearman_corr": spearman_corr,
                "distance_spl_pearson_corr": pearson_corr,
                "log_distance_spl_slope": log_distance_spl_slope,
                "inverse_square_fit_r2_proxy": inverse_square_fit_r2_proxy,
                "windowed_inverse_square_fit_r2_proxy": window_r2_proxy,
                "windowed_inverse_square_fit_start_sec": window_r2_start_sec,
                "windowed_inverse_square_fit_end_sec": window_r2_end_sec,
                "windowed_inverse_square_fit_best_n": window_r2_best_n,
                "legacy_windowed_r2": legacy_window_r2_proxy,
                "legacy_windowed_start_sec": legacy_window_r2_start_sec,
                "legacy_windowed_end_sec": legacy_window_r2_end_sec,
                "legacy_windowed_best_n": legacy_window_r2_best_n,
                "legacy_approaching_applicable": legacy_approaching_applicable,
                "legacy_approaching_start_sec": legacy_approaching_start_sec,
                "legacy_approaching_end_sec": legacy_approaching_end_sec,
                "legacy_approaching_distance_slope": legacy_approaching_distance_slope,
                "legacy_approaching_spl_slope": legacy_approaching_spl_slope,
                "legacy_approaching_distance_spl_corr": legacy_approaching_corr,
                "legacy_approaching_score": legacy_approaching_score,
                "legacy_lateral_applicable": legacy_lateral_applicable,
                "legacy_lateral_start_sec": legacy_lateral_start_sec,
                "legacy_lateral_end_sec": legacy_lateral_end_sec,
                "legacy_lateral_distance_cv": legacy_lateral_distance_cv,
                "legacy_lateral_bbox_motion": legacy_lateral_bbox_motion,
                "legacy_lateral_spl_variance_db": legacy_lateral_spl_variance_db,
                "legacy_lateral_spl_drift_db": legacy_lateral_spl_drift_db,
                "legacy_lateral_stability_score": legacy_lateral_stability_score,
                "legacy_strict_r2": legacy_strict_r2,
                "legacy_strict_best_n": legacy_strict_best_n,
                "legacy_strict_frames": legacy_strict_frames,
                "legacy_strict_start_sec": legacy_strict_start_sec,
                "legacy_strict_end_sec": legacy_strict_end_sec,
                "distance_spl_lag_sec": distance_spl_lag_sec,
                "valid_aligned_frame_ratio": valid_aligned_frame_ratio,
                "distance_dynamic_range": dist_range,
                "spl_dynamic_range_db": spl_range,
                "spl_peak_to_trough_db": spl_peak_to_trough_db,
                "approaching_segment_start_sec": approaching_segment_start_sec,
                "approaching_segment_end_sec": approaching_segment_end_sec,
                "distance_trend_slope": d_slope,
                "spl_trend_slope": s_slope,
                "approaching_distance_loudness_corr": approaching_corr,
                "approaching_consistency_score": approaching_consistency_score,
                "negative_distance_loudness_correlation": float(-approaching_corr) if np.isfinite(approaching_corr) else float("nan"),
                "monotonic_agreement_ratio": monotonic_agreement_ratio,
                "lateral_segment_start_sec": lateral_segment_start_sec,
                "lateral_segment_end_sec": lateral_segment_end_sec,
                "distance_cv": distance_cv,
                "bbox_motion_magnitude": bbox_motion_magnitude,
                "spl_variance_db": spl_variance_db,
                "spl_drift_db": spl_drift_db,
                "loudness_stability_score": loudness_stability_score,
                "lateral_distance_stability_score": lateral_distance_stability_score,
                "distance_proxy_orientation": "flipped" if np.nanmean(dist_i) != np.nanmean(raw_dist_i) else "raw",
            }
            out_rows.append(row_out)
            with open(out_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(row_out, ensure_ascii=False) + "\n")
        except Exception as e:
            out_rows.append({"sample_id": sample_id, "video_id": clean_text(row.get("video_id"), ""), "chunk_id": clean_text(row.get("chunk_id"), ""), "evaluation_status": "failed", "skip_reason": str(e), "applicable_testpoints": ""})

    fieldnames = list(out_rows[0].keys()) if out_rows else []
    write_csv_dicts(output_root / "receiver_observer_metrics.csv", out_rows, fieldnames)
    print(json.dumps({"csv": str(output_root / "receiver_observer_metrics.csv"), "jsonl": str(out_jsonl), "rows": len(out_rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
