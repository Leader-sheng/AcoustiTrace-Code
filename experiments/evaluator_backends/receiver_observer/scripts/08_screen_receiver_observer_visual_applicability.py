from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from common import ensure_dir, load_yaml


DEFAULT_CONFIG = SCRIPT_DIR.parent / "configs" / "receiver_observer_unified_v2_config.yaml"

SCORE_COLUMNS = {
    "distance": "windowed_inverse_square_fit_r2_proxy",
    "approaching": "approaching_consistency_score",
    "lateral": "loudness_stability_score",
}

APPLICABLE = "applicable"
NOT_APPLICABLE = "not_applicable"
UNVERIFIABLE = "unverifiable"


@dataclass
class TrackData:
    path: Path
    times: np.ndarray
    depth: np.ndarray
    bbox_x: np.ndarray
    bbox_y: np.ndarray
    bbox_area: np.ndarray
    valid_mask_ratio: np.ndarray
    depth_valid_ratio: np.ndarray
    source_visible_ratio: np.ndarray
    depth_field: str


def bool_enabled(value: object, default: bool = True) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def finite_float(value: object, default: float = float("nan")) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def cfg_float(cfg: dict, key: str, default: float) -> float:
    return finite_float(cfg.get(key, default), default)


def cfg_int(cfg: dict, key: str, default: int) -> int:
    try:
        return int(cfg.get(key, default))
    except Exception:
        return default


def nanmean_safe(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def nanmedian_safe(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.median(arr))


def robust_mad(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    med = np.median(arr)
    return float(np.median(np.abs(arr - med)))


def rolling_median_by_time(times: np.ndarray, values: np.ndarray, window_sec: float) -> np.ndarray:
    if len(values) < 3 or not np.isfinite(window_sec) or window_sec <= 0:
        return values.astype(float).copy()
    half = 0.5 * float(window_sec)
    out = np.asarray(values, dtype=float).copy()
    for i, t in enumerate(times):
        m = np.isfinite(values) & (times >= t - half) & (times <= t + half)
        if m.sum() >= 2:
            out[i] = float(np.nanmedian(values[m]))
    return out


def fit_line(times: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    m = np.isfinite(times) & np.isfinite(values)
    if m.sum() < 3:
        return float("nan"), float("nan")
    x = np.asarray(times[m], dtype=float)
    y = np.asarray(values[m], dtype=float)
    if np.nanmax(x) <= np.nanmin(x) or np.nanstd(y) < 1e-12:
        return float("nan"), float("nan")
    try:
        slope, intercept = np.polyfit(x, y, 1)
        pred = slope * x + intercept
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        return float(slope), float(r2)
    except Exception:
        return float("nan"), float("nan")


def first_existing_track(row: pd.Series, track_root: Path | None) -> Path | None:
    sample_id = str(row.get("sample_id", "")).strip()
    candidates: list[Path] = []
    if track_root is not None and sample_id:
        candidates.append(track_root / sample_id / "track.npz")
    batch_dir = str(row.get("batch_dir", "")).strip()
    if batch_dir and sample_id:
        candidates.append(Path(batch_dir) / "tracks" / sample_id / "track.npz")
    for key in ("track_path", "npz_path"):
        value = str(row.get(key, "")).strip()
        if value:
            candidates.append(Path(value))
    for path in candidates:
        if path.exists():
            return path
    return candidates[0] if candidates else None


def array_from_npz(data: np.lib.npyio.NpzFile, key: str, n: int, default: float = float("nan")) -> np.ndarray:
    if key in data:
        return np.asarray(data[key], dtype=float)
    return np.full(n, default, dtype=float)


def load_track(path: Path, app_cfg: dict) -> TrackData:
    data = np.load(path)
    keys = set(data.keys())
    if "frame_times_sec" not in keys:
        raise RuntimeError("missing_frame_times_sec")
    depth_candidates = [
        str(app_cfg.get("depth_field", "source_depth_proxy_smooth")),
        "source_depth_proxy_smooth",
        "source_depth_proxy",
        "source_depth_median",
        "source_depth_min",
        "source_depth_mean",
    ]
    depth_key = next((key for key in depth_candidates if key in keys), "")
    if not depth_key:
        raise RuntimeError("missing_depth_proxy")
    arrays = [np.asarray(data["frame_times_sec"], dtype=float), np.asarray(data[depth_key], dtype=float)]
    optional_keys = [
        "bbox_center_x",
        "bbox_center_y",
        "bbox_area_ratio",
        "valid_mask_ratio",
        "depth_valid_ratio",
        "source_visible_ratio",
    ]
    for key in optional_keys:
        if key in keys:
            arrays.append(np.asarray(data[key], dtype=float))
    n = min(len(a) for a in arrays if a.ndim == 1)
    if n <= 0:
        raise RuntimeError("empty_track_arrays")
    times = np.asarray(data["frame_times_sec"], dtype=float)[:n]
    depth = np.asarray(data[depth_key], dtype=float)[:n]
    bbox_x = array_from_npz(data, "bbox_center_x", n)[:n]
    bbox_y = array_from_npz(data, "bbox_center_y", n)[:n]
    bbox_area = array_from_npz(data, "bbox_area_ratio", n)[:n]
    valid_mask_ratio = array_from_npz(data, "valid_mask_ratio", n, 1.0)[:n]
    depth_valid_ratio = array_from_npz(data, "depth_valid_ratio", n, 1.0)[:n]
    source_visible_ratio = array_from_npz(data, "source_visible_ratio", n, 1.0)[:n]
    order = np.argsort(times)
    return TrackData(
        path=path,
        times=times[order],
        depth=depth[order],
        bbox_x=bbox_x[order],
        bbox_y=bbox_y[order],
        bbox_area=bbox_area[order],
        valid_mask_ratio=valid_mask_ratio[order],
        depth_valid_ratio=depth_valid_ratio[order],
        source_visible_ratio=source_visible_ratio[order],
        depth_field=depth_key,
    )


def split_reliable_runs(track: TrackData, app_cfg: dict) -> tuple[list[np.ndarray], list[float], list[str]]:
    times = track.times
    depth = track.depth
    n = len(times)
    min_valid_mask = cfg_float(app_cfg, "per_frame_min_valid_mask_ratio", 0.0)
    min_visible = cfg_float(app_cfg, "per_frame_min_source_visible_ratio", 0.0)
    max_gap = cfg_float(app_cfg, "max_tracking_gap_sec", 0.35)
    max_bbox_jump = cfg_float(app_cfg, "max_bbox_center_jump", 0.35)
    max_area_jump = cfg_float(app_cfg, "max_bbox_area_log_jump", 1.4)
    max_depth_jump_abs = cfg_float(app_cfg, "max_depth_jump_abs", 0.75)
    max_depth_jump_mad_factor = cfg_float(app_cfg, "max_depth_jump_mad_factor", 10.0)

    valid = np.isfinite(times) & np.isfinite(depth)
    valid &= np.asarray(track.valid_mask_ratio >= min_valid_mask, dtype=bool)
    valid &= np.asarray(track.source_visible_ratio >= min_visible, dtype=bool)

    depth_diffs = np.abs(np.diff(depth[np.isfinite(depth)]))
    mad = robust_mad(depth_diffs)
    if not np.isfinite(mad) or mad <= 1e-9:
        depth_jump_threshold = max_depth_jump_abs
    else:
        depth_jump_threshold = max(max_depth_jump_abs, max_depth_jump_mad_factor * mad)

    break_before = np.zeros(n, dtype=bool)
    gap_times: list[float] = []
    break_reasons: list[str] = []
    for i in range(1, n):
        if not valid[i - 1] or not valid[i]:
            break_before[i] = True
            continue
        dt = times[i] - times[i - 1]
        if not np.isfinite(dt) or dt <= 0 or dt > max_gap:
            break_before[i] = True
            gap_times.append(float(times[i]))
            break_reasons.append("tracking_gap")
            continue
        if np.isfinite(track.bbox_x[i]) and np.isfinite(track.bbox_x[i - 1]) and np.isfinite(track.bbox_y[i]) and np.isfinite(track.bbox_y[i - 1]):
            center_jump = math.hypot(float(track.bbox_x[i] - track.bbox_x[i - 1]), float(track.bbox_y[i] - track.bbox_y[i - 1]))
            if center_jump > max_bbox_jump:
                break_before[i] = True
                gap_times.append(float(times[i]))
                break_reasons.append("bbox_center_jump")
                continue
        if np.isfinite(track.bbox_area[i]) and np.isfinite(track.bbox_area[i - 1]) and track.bbox_area[i] > 0 and track.bbox_area[i - 1] > 0:
            area_jump = abs(math.log(float(track.bbox_area[i] / track.bbox_area[i - 1])))
            if area_jump > max_area_jump:
                break_before[i] = True
                gap_times.append(float(times[i]))
                break_reasons.append("bbox_area_jump")
                continue
        if abs(float(depth[i] - depth[i - 1])) > depth_jump_threshold:
            break_before[i] = True
            gap_times.append(float(times[i]))
            break_reasons.append("depth_jump")

    runs: list[np.ndarray] = []
    start = None
    for i in range(n):
        if not valid[i]:
            if start is not None and i - start >= 2:
                runs.append(np.arange(start, i, dtype=int))
            start = None
            continue
        if start is None or break_before[i]:
            if start is not None and i - start >= 2:
                runs.append(np.arange(start, i, dtype=int))
            start = i
    if start is not None and n - start >= 2:
        runs.append(np.arange(start, n, dtype=int))
    return runs, gap_times, break_reasons


def track_quality(track: TrackData, app_cfg: dict, runs: list[np.ndarray]) -> tuple[bool, str, dict[str, float]]:
    times = track.times
    finite = np.isfinite(times) & np.isfinite(track.depth)
    total_points = len(times)
    finite_points = int(finite.sum())
    duration = float(np.nanmax(times[finite]) - np.nanmin(times[finite])) if finite_points >= 2 else 0.0
    run_durations = [
        float(times[idx[-1]] - times[idx[0]])
        for idx in runs
        if len(idx) >= 2 and np.isfinite(times[idx[-1]]) and np.isfinite(times[idx[0]])
    ]
    longest_run = max(run_durations) if run_durations else 0.0
    stats = {
        "track_num_points": float(total_points),
        "track_finite_points": float(finite_points),
        "track_duration_sec": duration,
        "longest_reliable_run_sec": longest_run,
        "mean_valid_mask_ratio": nanmean_safe(track.valid_mask_ratio),
        "mean_depth_valid_ratio": nanmean_safe(track.depth_valid_ratio),
        "mean_source_visible_ratio": nanmean_safe(track.source_visible_ratio),
    }
    if finite_points < cfg_int(app_cfg, "min_valid_points", 6):
        return False, "too_few_finite_track_points", stats
    if duration < cfg_float(app_cfg, "min_track_duration_sec", 0.5):
        return False, "track_too_short", stats
    if finite_points / max(1, total_points) < cfg_float(app_cfg, "min_valid_track_ratio", 0.5):
        return False, "low_finite_track_ratio", stats
    if longest_run < cfg_float(app_cfg, "min_reliable_run_sec", 0.5):
        return False, "no_reliable_continuous_run", stats
    min_valid_mask = cfg_float(app_cfg, "min_mean_valid_mask_ratio", 0.0)
    min_depth_valid = cfg_float(app_cfg, "min_mean_depth_valid_ratio", 0.0)
    min_visible = cfg_float(app_cfg, "min_mean_source_visible_ratio", 0.0)
    if np.isfinite(stats["mean_valid_mask_ratio"]) and stats["mean_valid_mask_ratio"] < min_valid_mask:
        return False, "low_valid_mask_ratio", stats
    if np.isfinite(stats["mean_depth_valid_ratio"]) and stats["mean_depth_valid_ratio"] < min_depth_valid:
        return False, "low_depth_valid_ratio", stats
    if np.isfinite(stats["mean_source_visible_ratio"]) and stats["mean_source_visible_ratio"] < min_visible:
        return False, "low_source_visible_ratio", stats
    return True, "reliable_track", stats


def motion_windows_for_run(
    run_id: int,
    run_idx: np.ndarray,
    times: np.ndarray,
    depth_smooth: np.ndarray,
    sign: int,
    min_duration: float,
    min_depth_change: float,
    min_slope: float,
    app_cfg: dict,
    min_r2_override: float | None = None,
) -> list[dict]:
    out: list[dict] = []
    if len(run_idx) < 3:
        return out
    t = times[run_idx]
    d = depth_smooth[run_idx]
    min_points = cfg_int(app_cfg, "min_window_points", 3)
    neutral_step = cfg_float(app_cfg, "neutral_depth_step", 0.01)
    min_directional_fraction = cfg_float(app_cfg, "min_directional_fraction", 0.65)
    min_active_fraction = cfg_float(app_cfg, "min_active_direction_fraction", 0.25)
    max_counter_fraction = cfg_float(app_cfg, "max_counter_direction_fraction", 0.35)
    min_r2 = cfg_float(app_cfg, "min_motion_trend_r2", -1.0) if min_r2_override is None else float(min_r2_override)

    for i in range(len(run_idx) - 1):
        for j in range(i + min_points - 1, len(run_idx)):
            duration = float(t[j] - t[i])
            if duration < min_duration:
                continue
            sub_t = t[i : j + 1]
            sub_d = d[i : j + 1]
            if np.isfinite(sub_t).sum() < min_points or np.isfinite(sub_d).sum() < min_points:
                continue
            signed_delta = sign * float(sub_d[-1] - sub_d[0])
            slope, r2 = fit_line(sub_t, sub_d)
            signed_slope = sign * slope if np.isfinite(slope) else float("nan")
            if signed_delta < min_depth_change:
                continue
            if not np.isfinite(signed_slope) or signed_slope < min_slope:
                continue
            if np.isfinite(r2) and r2 < min_r2:
                continue
            diffs = sign * np.diff(sub_d)
            if len(diffs) == 0:
                continue
            directional_fraction = float(np.mean(diffs >= -neutral_step))
            active_fraction = float(np.mean(diffs > neutral_step))
            counter_fraction = float(np.mean(diffs < -neutral_step))
            if directional_fraction < min_directional_fraction:
                continue
            if active_fraction < min_active_fraction:
                continue
            if counter_fraction > max_counter_fraction:
                continue
            out.append(
                {
                    "run_id": int(run_id),
                    "start_sec": float(sub_t[0]),
                    "end_sec": float(sub_t[-1]),
                    "duration_sec": duration,
                    "depth_change": signed_delta,
                    "slope": float(slope),
                    "signed_slope": float(signed_slope),
                    "r2": float(r2),
                    "directional_fraction": directional_fraction,
                    "active_direction_fraction": active_fraction,
                    "counter_direction_fraction": counter_fraction,
                    "num_points": int(len(sub_t)),
                }
            )
    out.sort(key=lambda x: (x["duration_sec"], x["depth_change"], x["signed_slope"]), reverse=True)
    return out


def bbox_motion_for_run(track: TrackData, run_idx: np.ndarray) -> tuple[float, float]:
    x = track.bbox_x[run_idx]
    y = track.bbox_y[run_idx]
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return float("nan"), float("nan")
    x = x[m]
    y = y[m]
    center_range = math.hypot(float(np.nanmax(x) - np.nanmin(x)), float(np.nanmax(y) - np.nanmin(y)))
    steps = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    path_length = float(np.nansum(steps))
    return float(center_range), path_length


def classify_steps(times: np.ndarray, depth_smooth: np.ndarray, runs: list[np.ndarray], app_cfg: dict) -> list[tuple[float, float, str]]:
    neutral_step = cfg_float(app_cfg, "neutral_depth_step", 0.01)
    intervals: list[tuple[float, float, str]] = []
    for run_idx in runs:
        for a, b in zip(run_idx[:-1], run_idx[1:]):
            delta = float(depth_smooth[b] - depth_smooth[a])
            if delta > neutral_step:
                label = "receding"
            elif delta < -neutral_step:
                label = "approaching"
            else:
                label = "neutral"
            intervals.append((float(times[a]), float(times[b]), label))
    return intervals


def evaluate_track(track: TrackData, app_cfg: dict) -> dict:
    runs, gap_times, break_reasons = split_reliable_runs(track, app_cfg)
    reliable, quality_reason, quality_stats = track_quality(track, app_cfg, runs)
    depth_smooth = rolling_median_by_time(track.times, track.depth, cfg_float(app_cfg, "smoothing_window_sec", 0.35))

    base = {
        "track_path": str(track.path),
        "track_reliable": bool(reliable),
        "track_reliability_reason": quality_reason,
        "track_depth_field": track.depth_field,
        "track_gap_count": int(len(gap_times)),
        "track_gap_times_sec": "|".join(f"{x:.3f}" for x in gap_times[:20]),
        "track_break_reasons": "|".join(break_reasons[:20]),
        **quality_stats,
    }
    if not reliable:
        base.update(
            {
                "distance_status": UNVERIFIABLE,
                "distance_reason": quality_reason,
                "approaching_status": UNVERIFIABLE,
                "approaching_reason": quality_reason,
                "lateral_status": UNVERIFIABLE,
                "lateral_reason": quality_reason,
                "longest_receding_duration_sec": 0.0,
                "longest_approaching_duration_sec": 0.0,
                "lateral_bbox_center_motion": float("nan"),
                "lateral_bbox_path_length": float("nan"),
                "motion_step_intervals": classify_steps(track.times, depth_smooth, runs, app_cfg),
                "best_receding": None,
                "best_approaching": None,
                "best_lateral_receding": None,
                "best_lateral_approaching": None,
            }
        )
        return base

    main_min_change = cfg_float(app_cfg, "min_motion_depth_change", 0.12)
    approaching_min_change = cfg_float(
        app_cfg, "approaching_min_motion_depth_change", 0.15
    )
    lateral_min_change = cfg_float(app_cfg, "lateral_min_motion_depth_change", 0.05)
    min_slope = cfg_float(app_cfg, "min_motion_slope", 0.01)
    lateral_min_r2 = cfg_float(app_cfg, "lateral_min_motion_trend_r2", cfg_float(app_cfg, "min_motion_trend_r2", -1.0))
    distance_duration = cfg_float(app_cfg, "distance_min_receding_duration_sec", 1.5)
    approaching_duration = cfg_float(app_cfg, "approaching_min_duration_sec", 1.5)
    lateral_receding_duration = cfg_float(app_cfg, "lateral_min_receding_duration_sec", 0.5)
    lateral_approaching_duration = cfg_float(app_cfg, "lateral_min_approaching_duration_sec", 0.5)

    receding: list[dict] = []
    approaching: list[dict] = []
    lateral_receding: list[dict] = []
    lateral_approaching: list[dict] = []
    run_motion: dict[int, tuple[float, float]] = {}
    for run_id, run_idx in enumerate(runs):
        if len(run_idx) < 3:
            continue
        receding.extend(
            motion_windows_for_run(run_id, run_idx, track.times, depth_smooth, +1, distance_duration, main_min_change, min_slope, app_cfg)
        )
        approaching.extend(
            motion_windows_for_run(run_id, run_idx, track.times, depth_smooth, -1, approaching_duration, approaching_min_change, min_slope, app_cfg)
        )
        lateral_receding.extend(
            motion_windows_for_run(
                run_id,
                run_idx,
                track.times,
                depth_smooth,
                +1,
                lateral_receding_duration,
                lateral_min_change,
                min_slope,
                app_cfg,
                lateral_min_r2,
            )
        )
        lateral_approaching.extend(
            motion_windows_for_run(
                run_id,
                run_idx,
                track.times,
                depth_smooth,
                -1,
                lateral_approaching_duration,
                lateral_min_change,
                min_slope,
                app_cfg,
                lateral_min_r2,
            )
        )
        run_motion[run_id] = bbox_motion_for_run(track, run_idx)

    receding.sort(key=lambda x: (x["duration_sec"], x["depth_change"], x["signed_slope"]), reverse=True)
    approaching.sort(key=lambda x: (x["duration_sec"], x["depth_change"], x["signed_slope"]), reverse=True)
    lateral_receding.sort(key=lambda x: (x["duration_sec"], x["depth_change"], x["signed_slope"]), reverse=True)
    lateral_approaching.sort(key=lambda x: (x["duration_sec"], x["depth_change"], x["signed_slope"]), reverse=True)
    best_receding = receding[0] if receding else None
    best_approaching = approaching[0] if approaching else None
    longest_receding = float(best_receding["duration_sec"]) if best_receding else 0.0
    longest_approaching = float(best_approaching["duration_sec"]) if best_approaching else 0.0
    dominance_ratio = cfg_float(app_cfg, "main_direction_dominance_ratio", 1.5)
    receding_depth_change = float(best_receding["depth_change"]) if best_receding else 0.0
    approaching_depth_change = float(best_approaching["depth_change"]) if best_approaching else 0.0
    if best_receding and best_approaching:
        receding_dominates = receding_depth_change >= dominance_ratio * max(approaching_depth_change, 1e-9)
        approaching_dominates = approaching_depth_change >= dominance_ratio * max(receding_depth_change, 1e-9)
    else:
        receding_dominates = bool(best_receding)
        approaching_dominates = bool(best_approaching)
    ambiguous_main_front_back = bool(best_receding and best_approaching and not receding_dominates and not approaching_dominates)
    distance_status = APPLICABLE if best_receding and receding_dominates else NOT_APPLICABLE
    if not best_receding:
        distance_reason = f"no reliable receding segment >= {distance_duration:.3f}s"
    elif not receding_dominates:
        distance_reason = (
            f"receding segment is not dominant enough: receding_change={receding_depth_change:.3f}, "
            f"approaching_change={approaching_depth_change:.3f}, dominance_ratio={dominance_ratio:.3f}"
        )
    else:
        distance_reason = f"reliable dominant receding segment {longest_receding:.3f}s"

    approaching_status = APPLICABLE if best_approaching and approaching_dominates else NOT_APPLICABLE
    if not best_approaching:
        approaching_reason = f"no reliable approaching segment >= {approaching_duration:.3f}s"
    elif not approaching_dominates:
        approaching_reason = (
            f"approaching segment is not dominant enough: approaching_change={approaching_depth_change:.3f}, "
            f"receding_change={receding_depth_change:.3f}, dominance_ratio={dominance_ratio:.3f}"
        )
    else:
        approaching_reason = f"reliable dominant approaching segment {longest_approaching:.3f}s"

    min_center_motion = cfg_float(app_cfg, "lateral_min_bbox_center_motion", 0.03)
    min_path_length = cfg_float(app_cfg, "lateral_min_bbox_path_length", 0.05)
    best_pair: tuple[dict, dict, float, float] | None = None
    for app in lateral_approaching:
        for rec in lateral_receding:
            if int(app["run_id"]) != int(rec["run_id"]):
                continue
            center_motion, path_length = run_motion.get(int(app["run_id"]), (float("nan"), float("nan")))
            if not np.isfinite(center_motion) or not np.isfinite(path_length):
                continue
            if center_motion < min_center_motion or path_length < min_path_length:
                continue
            candidate = (app, rec, center_motion, path_length)
            if best_pair is None:
                best_pair = candidate
            else:
                best_score = best_pair[0]["duration_sec"] + best_pair[1]["duration_sec"]
                cand_score = app["duration_sec"] + rec["duration_sec"]
                if cand_score > best_score:
                    best_pair = candidate

    if ambiguous_main_front_back:
        lateral_status = NOT_APPLICABLE
        lateral_reason = (
            f"ambiguous strong front-back motion: receding_change={receding_depth_change:.3f}, "
            f"approaching_change={approaching_depth_change:.3f}"
        )
        lateral_bbox_center_motion = float("nan")
        lateral_bbox_path_length = float("nan")
        best_lat_app = None
        best_lat_rec = None
    elif best_pair is None:
        same_run_possible = any(int(a["run_id"]) == int(r["run_id"]) for a in lateral_approaching for r in lateral_receding)
        if not lateral_approaching or not lateral_receding:
            lateral_reason = (
                f"needs both approaching >= {lateral_approaching_duration:.3f}s and receding >= {lateral_receding_duration:.3f}s"
            )
        elif not same_run_possible:
            lateral_reason = "approaching and receding segments are not in the same reliable run"
        else:
            best_motion = max((run_motion.get(int(a["run_id"]), (float("nan"), float("nan"))) for a in lateral_approaching), default=(float("nan"), float("nan")))
            lateral_reason = (
                f"insufficient image-plane motion: center={best_motion[0]:.4f}, path={best_motion[1]:.4f}"
            )
        lateral_status = NOT_APPLICABLE
        lateral_bbox_center_motion = float("nan")
        lateral_bbox_path_length = float("nan")
        best_lat_app = None
        best_lat_rec = None
    else:
        best_lat_app, best_lat_rec, lateral_bbox_center_motion, lateral_bbox_path_length = best_pair
        lateral_status = APPLICABLE
        lateral_reason = (
            f"same reliable run has approaching {best_lat_app['duration_sec']:.3f}s, "
            f"receding {best_lat_rec['duration_sec']:.3f}s, bbox center motion {lateral_bbox_center_motion:.4f}"
        )

    base.update(
        {
            "distance_status": distance_status,
            "distance_reason": distance_reason,
            "approaching_status": approaching_status,
            "approaching_reason": approaching_reason,
            "lateral_status": lateral_status,
            "lateral_reason": lateral_reason,
            "longest_receding_duration_sec": longest_receding,
            "longest_approaching_duration_sec": longest_approaching,
            "lateral_bbox_center_motion": lateral_bbox_center_motion,
            "lateral_bbox_path_length": lateral_bbox_path_length,
            "motion_step_intervals": classify_steps(track.times, depth_smooth, runs, app_cfg),
            "best_receding": best_receding,
            "best_approaching": best_approaching,
            "best_lateral_receding": best_lat_rec,
            "best_lateral_approaching": best_lat_app,
            "depth_smooth": depth_smooth,
        }
    )
    return base


def compact_motion(motion: dict | None, prefix: str) -> dict:
    keys = ["start_sec", "end_sec", "duration_sec", "depth_change", "slope", "r2", "directional_fraction", "counter_direction_fraction", "run_id"]
    out = {}
    for key in keys:
        out[f"{prefix}_{key}"] = motion.get(key, float("nan")) if motion else float("nan")
    return out


def make_applicability_row(row: pd.Series, result: dict, sample_id: str, video_path: str) -> dict:
    out = {
        "sample_id": sample_id,
        "video_path": video_path,
        "track_path": result.get("track_path", ""),
        "track_reliable": result.get("track_reliable", False),
        "track_reliability_reason": result.get("track_reliability_reason", ""),
        "track_depth_field": result.get("track_depth_field", ""),
        "track_num_points": result.get("track_num_points", ""),
        "track_finite_points": result.get("track_finite_points", ""),
        "track_duration_sec": result.get("track_duration_sec", ""),
        "longest_reliable_run_sec": result.get("longest_reliable_run_sec", ""),
        "track_gap_count": result.get("track_gap_count", ""),
        "track_gap_times_sec": result.get("track_gap_times_sec", ""),
        "track_break_reasons": result.get("track_break_reasons", ""),
        "mean_valid_mask_ratio": result.get("mean_valid_mask_ratio", ""),
        "mean_depth_valid_ratio": result.get("mean_depth_valid_ratio", ""),
        "mean_source_visible_ratio": result.get("mean_source_visible_ratio", ""),
        "longest_receding_duration_sec": result.get("longest_receding_duration_sec", 0.0),
        "longest_approaching_duration_sec": result.get("longest_approaching_duration_sec", 0.0),
        "lateral_bbox_center_motion": result.get("lateral_bbox_center_motion", ""),
        "lateral_bbox_path_length": result.get("lateral_bbox_path_length", ""),
        "distance_attenuation_applicability_status": result.get("distance_status", UNVERIFIABLE),
        "distance_attenuation_applicable": result.get("distance_status") == APPLICABLE,
        "distance_attenuation_applicability_reason": result.get("distance_reason", ""),
        "approaching_enhancement_applicability_status": result.get("approaching_status", UNVERIFIABLE),
        "approaching_enhancement_applicable": result.get("approaching_status") == APPLICABLE,
        "approaching_enhancement_applicability_reason": result.get("approaching_reason", ""),
        "lateral_loudness_stability_applicability_status": result.get("lateral_status", UNVERIFIABLE),
        "lateral_loudness_stability_applicable": result.get("lateral_status") == APPLICABLE,
        "lateral_loudness_stability_applicability_reason": result.get("lateral_reason", ""),
    }
    out.update(compact_motion(result.get("best_receding"), "best_receding"))
    out.update(compact_motion(result.get("best_approaching"), "best_approaching"))
    out.update(compact_motion(result.get("best_lateral_receding"), "best_lateral_receding"))
    out.update(compact_motion(result.get("best_lateral_approaching"), "best_lateral_approaching"))
    return out


def plot_debug(track: TrackData, result: dict, out_path: Path, sample_id: str, video_path: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] Cannot import matplotlib for debug plot: {exc}", file=sys.stderr)
        return

    depth_smooth = result.get("depth_smooth")
    if depth_smooth is None:
        depth_smooth = rolling_median_by_time(track.times, track.depth, 0.35)
    intervals = result.get("motion_step_intervals", [])
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    ax = axes[0]
    for start, end, label in intervals:
        color = {"approaching": "#D1495B", "receding": "#2E86AB", "neutral": "#A0A0A0"}.get(label, "#A0A0A0")
        ax.axvspan(start, end, color=color, alpha=0.08, linewidth=0)
    ax.plot(track.times, track.depth, color="#999999", linewidth=1.0, label="depth proxy raw")
    ax.plot(track.times, depth_smooth, color="#111111", linewidth=1.8, label="depth proxy smoothed")
    for motion, color, name in [
        (result.get("best_receding"), "#2E86AB", "best receding"),
        (result.get("best_approaching"), "#D1495B", "best approaching"),
    ]:
        if motion:
            ax.axvspan(float(motion["start_sec"]), float(motion["end_sec"]), color=color, alpha=0.22, label=name)
    for token in str(result.get("track_gap_times_sec", "")).split("|"):
        if token:
            ax.axvline(float(token), color="#000000", linestyle="--", linewidth=0.8, alpha=0.35)
    ax.set_ylabel("depth proxy")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(track.times, track.bbox_x, label="bbox_center_x", color="#2A9D8F")
    ax.plot(track.times, track.bbox_y, label="bbox_center_y", color="#E9C46A")
    ax.set_ylabel("bbox center")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[2]
    ax.plot(track.times, track.bbox_area, label="bbox_area_ratio", color="#6A4C93")
    ax.plot(track.times, track.source_visible_ratio, label="source_visible_ratio", color="#F4A261")
    ax.plot(track.times, track.valid_mask_ratio, label="valid_mask_ratio", color="#264653")
    ax.set_ylabel("area / quality")
    ax.set_xlabel("time (sec)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.25)

    title = (
        f"{sample_id}\n"
        f"distance={result.get('distance_status')} | approaching={result.get('approaching_status')} | "
        f"lateral={result.get('lateral_status')}\n{video_path}"
    )
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def parse_sample_ids(text: str) -> set[str]:
    if not text:
        return set()
    return {x.strip() for x in text.replace("\n", ",").split(",") if x.strip()}


def parse_video_paths_file(path: str) -> set[str]:
    if not path:
        return set()
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return {line.strip() for line in lines if line.strip()}


def should_keep_row(row: pd.Series, sample_ids: set[str], video_paths: set[str]) -> bool:
    if sample_ids and str(row.get("sample_id", "")).strip() not in sample_ids:
        return False
    if video_paths:
        video = str(row.get("video_path", row.get("event_clip_path", ""))).strip()
        if video not in video_paths:
            return False
    return True


def apply_screening_to_row(out: pd.DataFrame, idx: int, app_row: dict) -> None:
    mapping = {
        "distance": (
            "distance_attenuation_applicability_status",
            "distance_attenuation_applicability_reason",
            "visual_distance_attenuation",
            "distance_decay_applicable",
        ),
        "approaching": (
            "approaching_enhancement_applicability_status",
            "approaching_enhancement_applicability_reason",
            "visual_approaching_enhancement",
            "approaching_applicable",
        ),
        "lateral": (
            "lateral_loudness_stability_applicability_status",
            "lateral_loudness_stability_applicability_reason",
            "visual_lateral_loudness_stability",
            "lateral_applicable",
        ),
    }
    applicable_testpoints: list[str] = []
    for key, score_col in SCORE_COLUMNS.items():
        status_col, reason_col, prefix, legacy_app_col = mapping[key]
        status = str(app_row.get(status_col, UNVERIFIABLE))
        reason = str(app_row.get(reason_col, ""))
        out.at[idx, f"{prefix}_applicability_status"] = status
        out.at[idx, f"{prefix}_applicability_reason"] = reason
        out.at[idx, f"{prefix}_applicable"] = status == APPLICABLE
        if legacy_app_col in out.columns:
            original_col = f"original_{legacy_app_col}"
            if original_col not in out.columns:
                out[original_col] = out[legacy_app_col]
            if out[legacy_app_col].dtype != object:
                out[legacy_app_col] = out[legacy_app_col].astype(object)
            out.at[idx, legacy_app_col] = bool(status == APPLICABLE)
        if score_col in out.columns:
            original_col = f"original_{score_col}"
            if original_col not in out.columns:
                out[original_col] = out[score_col]
            if status != APPLICABLE:
                out.at[idx, score_col] = np.nan
        if status == APPLICABLE:
            applicable_testpoints.append(
                {
                    "distance": "distance_decay",
                    "approaching": "approaching",
                    "lateral": "lateral_motion_loudness_stability",
                }[key]
            )
    out.at[idx, "visual_applicability_any_applicable"] = bool(applicable_testpoints)
    out.at[idx, "visual_applicability_unverifiable_any"] = any(
        str(app_row.get(col, "")) == UNVERIFIABLE
        for col in [
            "distance_attenuation_applicability_status",
            "approaching_enhancement_applicability_status",
            "lateral_loudness_stability_applicability_status",
        ]
    )
    if "applicable_testpoints" in out.columns and out["applicable_testpoints"].dtype != object:
        out["applicable_testpoints"] = out["applicable_testpoints"].astype(object)
    out.at[idx, "visual_applicable_testpoints"] = "|".join(applicable_testpoints)
    out.at[idx, "applicable_testpoints"] = "|".join(applicable_testpoints)


def summarize(app_rows: pd.DataFrame, screened: pd.DataFrame) -> dict:
    summary: dict[str, object] = {"rows": int(len(app_rows))}
    specs = {
        "distance": ("distance_attenuation_applicability_status", SCORE_COLUMNS["distance"]),
        "approaching": ("approaching_enhancement_applicability_status", SCORE_COLUMNS["approaching"]),
        "lateral": ("lateral_loudness_stability_applicability_status", SCORE_COLUMNS["lateral"]),
    }
    for key, (status_col, score_col) in specs.items():
        status = app_rows[status_col].astype(str) if status_col in app_rows else pd.Series(dtype=str)
        final_valid = pd.to_numeric(screened.get(score_col, pd.Series(dtype=float)), errors="coerce").notna()
        summary[key] = {
            "raw_candidate_count": int(len(app_rows)),
            "visual_applicable_count": int((status == APPLICABLE).sum()),
            "visual_not_applicable_count": int((status == NOT_APPLICABLE).sum()),
            "visual_unverifiable_count": int((status == UNVERIFIABLE).sum()),
            "final_valid_score_count": int(final_valid.sum()),
        }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Screen receiver-observer metrics by visual motion applicability from track.npz.")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--metrics_csv", required=True)
    ap.add_argument("--track_root", default="")
    ap.add_argument("--out_csv", default="")
    ap.add_argument("--applicability_csv", default="")
    ap.add_argument("--summary_json", default="")
    ap.add_argument("--debug_dir", default="")
    ap.add_argument("--sample-ids", default="")
    ap.add_argument("--video-paths-file", default="")
    ap.add_argument("--no-debug-plots", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    app_cfg = dict(cfg.get("visual_applicability", {}))
    if not bool_enabled(app_cfg.get("enabled", True), True):
        print(json.dumps({"status": "disabled", "metrics_csv": args.metrics_csv}, indent=2))
        return

    metrics_path = Path(args.metrics_csv)
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    out_csv = Path(args.out_csv or metrics_path.with_name("receiver_observer_unified_v2_screened_metrics.csv"))
    applicability_csv = Path(args.applicability_csv or metrics_path.with_name("receiver_observer_visual_applicability.csv"))
    summary_json = Path(args.summary_json or metrics_path.with_name("receiver_observer_visual_applicability_summary.json"))
    debug_dir = Path(args.debug_dir or metrics_path.parent / "visual_applicability_debug")
    track_root = Path(args.track_root) if args.track_root else None
    sample_ids = parse_sample_ids(args.sample_ids)
    video_paths = parse_video_paths_file(args.video_paths_file)

    source = pd.read_csv(metrics_path)
    selected_mask = source.apply(lambda row: should_keep_row(row, sample_ids, video_paths), axis=1)
    if sample_ids or video_paths:
        source = source[selected_mask].copy().reset_index(drop=True)
    screened = source.copy()

    app_rows: list[dict] = []
    debug_payload: list[dict] = []
    for idx, row in screened.iterrows():
        sample_id = str(row.get("sample_id", "")).strip()
        video_path = str(row.get("video_path", row.get("event_clip_path", ""))).strip()
        track_path = first_existing_track(row, track_root)
        if track_path is None or not track_path.exists():
            result = {
                "track_path": str(track_path or ""),
                "track_reliable": False,
                "track_reliability_reason": "missing_track_npz",
                "track_depth_field": "",
                "distance_status": UNVERIFIABLE,
                "distance_reason": "missing_track_npz",
                "approaching_status": UNVERIFIABLE,
                "approaching_reason": "missing_track_npz",
                "lateral_status": UNVERIFIABLE,
                "lateral_reason": "missing_track_npz",
                "longest_receding_duration_sec": 0.0,
                "longest_approaching_duration_sec": 0.0,
                "lateral_bbox_center_motion": float("nan"),
                "lateral_bbox_path_length": float("nan"),
            }
            app_row = make_applicability_row(row, result, sample_id, video_path)
            app_rows.append(app_row)
            apply_screening_to_row(screened, idx, app_row)
            debug_payload.append(app_row)
            continue
        try:
            track = load_track(track_path, app_cfg)
            result = evaluate_track(track, app_cfg)
            app_row = make_applicability_row(row, result, sample_id, video_path)
            app_rows.append(app_row)
            apply_screening_to_row(screened, idx, app_row)
            debug_payload.append(app_row)
            if not args.no_debug_plots:
                plot_debug(track, result, debug_dir / f"{sample_id}_visual_applicability.png", sample_id, video_path)
        except Exception as exc:
            result = {
                "track_path": str(track_path),
                "track_reliable": False,
                "track_reliability_reason": f"track_processing_failed:{exc}",
                "track_depth_field": "",
                "distance_status": UNVERIFIABLE,
                "distance_reason": f"track_processing_failed:{exc}",
                "approaching_status": UNVERIFIABLE,
                "approaching_reason": f"track_processing_failed:{exc}",
                "lateral_status": UNVERIFIABLE,
                "lateral_reason": f"track_processing_failed:{exc}",
                "longest_receding_duration_sec": 0.0,
                "longest_approaching_duration_sec": 0.0,
                "lateral_bbox_center_motion": float("nan"),
                "lateral_bbox_path_length": float("nan"),
            }
            app_row = make_applicability_row(row, result, sample_id, video_path)
            app_rows.append(app_row)
            apply_screening_to_row(screened, idx, app_row)
            debug_payload.append(app_row)

    app_frame = pd.DataFrame(app_rows)
    ensure_dir(out_csv.parent)
    ensure_dir(applicability_csv.parent)
    screened.to_csv(out_csv, index=False)
    app_frame.to_csv(applicability_csv, index=False)
    summary = summarize(app_frame, screened)
    summary.update(
        {
            "metrics_csv": str(metrics_path),
            "screened_metrics_csv": str(out_csv),
            "applicability_csv": str(applicability_csv),
            "debug_dir": str(debug_dir),
            "track_root": str(track_root or ""),
        }
    )
    ensure_dir(summary_json.parent)
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
