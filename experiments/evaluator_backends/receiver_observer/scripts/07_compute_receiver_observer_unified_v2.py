from __future__ import annotations

"""Compatibility exporter for intermediate receiver/observer tables.

Paper-reported validity labels are assigned by
08_screen_receiver_observer_visual_applicability.py. Applicability fields
written here are pre-screen candidates only.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from common import ensure_dir, load_yaml


DEFAULT_CONFIG = SCRIPT_DIR.parent / "configs" / "receiver_observer_unified_v2_config.yaml"
OUT_FIELDS = [
    "sample_id", "video_path", "split", "evaluation_status", "applicable_testpoints",
    "distance_decay_applicable", "approaching_applicable", "equal_distance_branch_applicable",
    "constant_distance_applicable", "lateral_applicable",
    "inverse_square_fit_r2_proxy", "windowed_inverse_square_fit_r2_proxy",
    "distance_spl_spearman_corr", "distance_spl_pearson_corr", "log_distance_spl_slope",
    "distance_spl_lag_sec", "approaching_consistency_score", "approaching_fit_r2_proxy",
    "equal_distance_num_pairs", "equal_distance_depth_overlap_range",
    "equal_distance_spl_mae_db", "equal_distance_spl_median_abs_error_db",
    "equal_distance_rms_log_error", "equal_distance_consistency_score",
    "monotonic_agreement_ratio", "loudness_stability_score", "constant_distance_stability_score",
    "lateral_distance_stability_score", "verified_lateral_motion",
    "spl_variance_db", "spl_drift_db", "rms_cv", "distance_cv", "bbox_motion_magnitude",
    "missing_reason", "error_msg", "feature_source",
]


def load_v1():
    spec = importlib.util.spec_from_file_location("receiver_metrics_v1", SCRIPT_DIR / "05_compute_receiver_observer_metrics.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


V1 = load_v1()


def bool_value(value) -> bool:
    return str(value).lower() in {"true", "1", "yes"}


def r2_fit(x: np.ndarray, y: np.ndarray) -> float:
    return V1.r2_proxy(x, y)


def branch_window_medians(dist: np.ndarray, spl: np.ndarray, rms: np.ndarray, times: np.ndarray, closest_time: float, window_sec: float, slide_sec: float, branch: str) -> list[dict]:
    rows = []
    if branch == "approaching":
        end = closest_time
        intervals = []
        while end - window_sec >= float(times[0]) - 1e-9:
            intervals.append((end - window_sec, end))
            end -= slide_sec
    else:
        start = closest_time
        intervals = []
        while start + window_sec <= float(times[-1]) + 1e-9:
            intervals.append((start, start + window_sec))
            start += slide_sec
    for start, end in intervals:
        w = (times >= start) & (times <= end)
        if w.sum() >= 3:
            rows.append({
                "center": 0.5 * (start + end),
                "dist": float(np.nanmedian(dist[w])),
                "spl": float(np.nanmedian(spl[w])),
                "rms": float(np.nanmedian(rms[w])),
            })
    return rows


def equal_distance_branch_metrics(dist: np.ndarray, spl: np.ndarray, rms: np.ndarray, times: np.ndarray, cfg: dict) -> dict | None:
    mc = cfg["metrics"]
    valid = np.isfinite(dist) & np.isfinite(spl) & np.isfinite(rms) & np.isfinite(times)
    if valid.sum() < 7:
        return None
    dist, spl, rms, times = dist[valid], spl[valid], rms[valid], times[valid]
    closest_idx = int(np.nanargmin(dist))
    if closest_idx < 2 or closest_idx > len(dist) - 3:
        return None
    closest_time = float(times[closest_idx])
    pair_window = float(mc.get("equal_distance_window_sec", mc["window_sec"]))
    pair_slide = float(mc.get("equal_distance_slide_sec", mc["slide_sec"]))
    approaching = branch_window_medians(dist, spl, rms, times, closest_time, pair_window, pair_slide, "approaching")
    receding = branch_window_medians(dist, spl, rms, times, closest_time, pair_window, pair_slide, "receding")
    if len(approaching) < 2 or len(receding) < 2:
        return None
    overlap_low = max(min(x["dist"] for x in approaching), min(x["dist"] for x in receding))
    overlap_high = min(max(x["dist"] for x in approaching), max(x["dist"] for x in receding))
    overlap_range = float(overlap_high - overlap_low)
    if overlap_range < float(mc["equal_distance_min_branch_range"]):
        return None
    total_range = float(np.nanmax(dist) - np.nanmin(dist))
    tolerance = max(1e-8, float(mc["equal_distance_depth_tolerance_ratio"]) * total_range)
    available = {
        i for i, row in enumerate(receding)
        if overlap_low <= row["dist"] <= overlap_high
    }
    pairs = []
    for left in sorted(approaching, key=lambda x: x["dist"]):
        if not (overlap_low <= left["dist"] <= overlap_high) or not available:
            continue
        best_idx = min(available, key=lambda i: abs(receding[i]["dist"] - left["dist"]))
        right = receding[best_idx]
        depth_error = abs(right["dist"] - left["dist"])
        if depth_error <= tolerance:
            pairs.append((left, right, depth_error))
            available.remove(best_idx)
    if len(pairs) < int(mc["equal_distance_min_pairs"]):
        return None
    spl_errors = np.asarray([abs(a["spl"] - b["spl"]) for a, b, _ in pairs], dtype=float)
    rms_log_errors = np.asarray([abs(np.log((b["rms"] + 1e-9) / (a["rms"] + 1e-9))) for a, b, _ in pairs], dtype=float)
    spl_median = float(np.nanmedian(spl_errors))
    rms_median = float(np.nanmedian(rms_log_errors))
    spl_score = float(np.exp(-spl_median / float(mc["equal_distance_spl_sigma_db"])))
    rms_score = float(np.exp(-rms_median / float(mc["equal_distance_rms_log_sigma"])))
    return {
        "equal_distance_num_pairs": len(pairs),
        "equal_distance_depth_overlap_range": overlap_range,
        "equal_distance_spl_mae_db": float(np.nanmean(spl_errors)),
        "equal_distance_spl_median_abs_error_db": spl_median,
        "equal_distance_rms_log_error": rms_median,
        "equal_distance_consistency_score": float(0.5 * spl_score + 0.5 * rms_score),
    }


def best_windows(dist: np.ndarray, spl: np.ndarray, rms: np.ndarray, bbox_x: np.ndarray, times: np.ndarray, cfg: dict) -> dict:
    mc = cfg["metrics"]
    window, slide = float(mc["window_sec"]), float(mc["slide_sec"])
    constant_distance_max_cv = float(mc.get("constant_distance_max_cv", mc["lateral_max_distance_cv"]))
    best_app = None
    best_constant = None
    start = float(times[0])
    while start + window <= float(times[-1]) + 1e-9:
        end = start + window
        w = (times >= start) & (times <= end)
        if w.sum() >= 3:
            wd, ws, wr, wx, wt = dist[w], spl[w], rms[w], bbox_x[w], times[w]
            d_range = float(np.nanmax(wd) - np.nanmin(wd))
            d_slope = V1.linfit_slope(wt, wd)
            s_slope = V1.linfit_slope(wt, ws)
            corr, _ = V1.corr_safe(wd, ws)
            monotonic = float(np.mean(np.sign(np.diff(wd)) * np.sign(np.diff(ws)) <= 0))
            if d_range >= float(mc["distance_decay_min_range"]) and d_slope < -float(mc["approaching_min_slope"]):
                fit = r2_fit(wd, ws)
                score = float(np.nanmean([max(0.0, -corr) if np.isfinite(corr) else 0.0, max(0.0, min(1.0, s_slope / 10.0)), monotonic]))
                if best_app is None or score > best_app["approaching_consistency_score"]:
                    best_app = {"approaching_consistency_score": score, "approaching_fit_r2_proxy": fit, "monotonic_agreement_ratio": monotonic}
            dcv = float(np.nanstd(wd) / (np.nanmean(np.abs(wd)) + 1e-8))
            motion = float(np.nanmax(wx) - np.nanmin(wx)) if len(wx) else np.nan
            # Constant-distance / lateral loudness stability: the main branch
            # is a locally stable depth window, with image-plane motion kept as
            # a diagnostic subset.
            if dcv <= constant_distance_max_cv:
                svar = float(np.nanvar(ws))
                drift = float(abs(V1.linfit_slope(wt, ws)) * (wt[-1] - wt[0]))
                rcv = float(np.nanstd(wr) / (np.nanmean(np.abs(wr)) + 1e-8))
                stability = float(1.0 / (1.0 + svar + abs(drift) / 10.0 + rcv))
                verified_lateral = bool(np.isfinite(motion) and motion >= float(mc["lateral_min_bbox_motion"]))
                if best_constant is None or stability > best_constant["constant_distance_stability_score"]:
                    best_constant = {
                        "distance_cv": dcv,
                        "bbox_motion_magnitude": motion,
                        "spl_variance_db": svar,
                        "spl_drift_db": drift,
                        "rms_cv": rcv,
                        "loudness_stability_score": stability,
                        "constant_distance_stability_score": stability,
                        "lateral_distance_stability_score": float(1.0 / (1.0 + dcv)),
                        "verified_lateral_motion": verified_lateral,
                    }
        start += slide
    return {"approaching": best_app, "constant_distance": best_constant}


def from_cache(row: pd.Series, args, cfg: dict) -> dict:
    sample_id = str(row.get("sample_id", ""))
    out = {k: "" for k in OUT_FIELDS}
    out.update({"sample_id": sample_id, "video_path": row.get("event_clip_path", row.get("video_path", "")), "split": args.split, "feature_source": "aligned_cache"})
    audio_npz = Path(args.audio_root) / sample_id / "audio_features.npz"
    track_npz = Path(args.track_root) / sample_id / "track.npz"
    if not audio_npz.exists() or not track_npz.exists():
        out.update({"evaluation_status": "failed", "missing_reason": "missing_audio_or_track", "error_msg": ""})
        return out
    try:
        audio = np.load(audio_npz)
        track = np.load(track_npz)
        at = np.asarray(audio["frame_times_sec"], dtype=float)
        spl = np.asarray(audio["spl_curve_db"], dtype=float)
        rms = np.asarray(audio["rms_curve"], dtype=float)
        tt = np.asarray(track["frame_times_sec"], dtype=float)
        dist = np.asarray(track["source_depth_proxy_smooth"], dtype=float)
        if "bbox_center_x" in track:
            bx = np.asarray(track["bbox_center_x"], dtype=float)
        else:
            bx = np.asarray(track["bbox_area_ratio"], dtype=float)
        common = np.linspace(max(at.min(), tt.min()), min(at.max(), tt.max()), min(len(at), len(tt)))
        di, si, ri, bi = np.interp(common, tt, dist), np.interp(common, at, spl), np.interp(common, at, rms), np.interp(common, tt, bx)
        dist_range = float(np.nanmax(di) - np.nanmin(di))
        distance_app = dist_range >= float(cfg["metrics"]["distance_decay_min_range"])
        logdist = np.log(np.maximum(di - np.nanmin(di) + 1e-6, 1e-6))
        spear, pear = V1.corr_safe(di, si)
        wr2, _, _, _ = V1.windowed_distance_r2_search(
            di,
            si,
            common,
            float(cfg["metrics"]["window_sec"]),
            float(cfg["metrics"]["slide_sec"]),
            float(cfg["metrics"]["distance_r2_search_min_n"]),
            float(cfg["metrics"]["distance_r2_search_max_n"]),
            int(cfg["metrics"]["distance_r2_search_density"]),
        )
        found = best_windows(di, si, ri, bi, common, cfg)
        app, constant = found["approaching"], found["constant_distance"]
        equal_distance = equal_distance_branch_metrics(di, si, ri, common, cfg)
        out.update({
            "evaluation_status": "success", "distance_decay_applicable": distance_app, "approaching_applicable": app is not None,
            "equal_distance_branch_applicable": equal_distance is not None,
            "constant_distance_applicable": constant is not None,
            # Intermediate constant-distance candidate; script 08 additionally
            # requires verified image-plane motion for final Lateral validity.
            "lateral_applicable": constant is not None,
            "inverse_square_fit_r2_proxy": r2_fit(logdist, si), "windowed_inverse_square_fit_r2_proxy": wr2,
            "distance_spl_spearman_corr": spear, "distance_spl_pearson_corr": pear, "log_distance_spl_slope": V1.linfit_slope(logdist, si),
            "distance_spl_lag_sec": V1.lag_proxy(di, si, common),
        })
        if app:
            out.update(app)
        if constant:
            out.update(constant)
        if equal_distance:
            out.update(equal_distance)
        out["applicable_testpoints"] = "|".join([n for n, ok in [
            ("distance_decay", distance_app),
            ("approaching", app is not None),
            ("constant_distance_loudness_stability", constant is not None),
            ("equal_distance_branch_consistency", equal_distance is not None),
        ] if ok])
    except Exception as exc:
        out.update({"evaluation_status": "failed", "missing_reason": "processing_failed", "error_msg": str(exc)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Unified Receiver / Observer v2 metric extraction for GT or generated videos.")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--manifest", default="")
    ap.add_argument("--audio_root", default="")
    ap.add_argument("--track_root", default="")
    ap.add_argument("--split", default="")
    ap.add_argument("--out_csv", default="")
    ap.add_argument("--run_id", default="unified_v2_default")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    split = args.split or cfg["data"].get("split", "gt")
    args.split = split
    output_root = Path(cfg["output"]["root"]) if args.run_id == "unified_v2_default" else SCRIPT_DIR.parent / "outputs_v2" / args.run_id
    out_csv = Path(args.out_csv or output_root / "metrics" / "receiver_observer_unified_v2_metrics.csv")
    manifest = pd.read_csv(args.manifest or cfg["data"]["manifest"])
    if args.limit > 0:
        manifest = manifest.head(args.limit)
    args.audio_root = args.audio_root or cfg["data"]["audio_root"]
    args.track_root = args.track_root or cfg["data"]["track_root"]
    result = pd.DataFrame(
        [from_cache(row, args, cfg) for _, row in manifest.iterrows()], columns=OUT_FIELDS
    )
    if args.dry_run:
        print({"rows": len(result), "output": str(out_csv), "split": split})
        return
    ensure_dir(out_csv.parent)
    result.to_csv(out_csv, index=False)
    print(json.dumps({"rows": len(result), "success": int((result["evaluation_status"] == "success").sum()), "distance_applicable": int(result["distance_decay_applicable"].astype(str).str.lower().isin(["true", "1"]).sum()), "approaching_applicable": int(result["approaching_applicable"].astype(str).str.lower().isin(["true", "1"]).sum()), "equal_distance_branch_applicable": int(result["equal_distance_branch_applicable"].astype(str).str.lower().isin(["true", "1"]).sum()), "constant_distance_auxiliary_applicable": int(result["constant_distance_applicable"].astype(str).str.lower().isin(["true", "1"]).sum()), "output": str(out_csv)}, indent=2))


if __name__ == "__main__":
    main()
