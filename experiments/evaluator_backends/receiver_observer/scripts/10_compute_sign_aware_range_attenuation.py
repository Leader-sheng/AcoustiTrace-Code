from __future__ import annotations

"""Compute the Range Attenuation score from aligned distance and SPL tracks."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SCORE_FIELD = "sign_aware_windowed_inverse_square_fit_r2_proxy"


def score_windows(
    distance: np.ndarray,
    spl_db: np.ndarray,
    times: np.ndarray,
    *,
    window_sec: float = 0.4,
    slide_sec: float = 0.05,
    min_exponent: float = 5.0,
    max_exponent: float = 50.0,
    exponent_count: int = 450,
) -> tuple[float, float, float, float, str]:
    valid = np.isfinite(distance) & np.isfinite(spl_db) & np.isfinite(times)
    if int(valid.sum()) < 4:
        return np.nan, np.nan, np.nan, np.nan, "insufficient_finite_points"
    distance = np.asarray(distance[valid], dtype=np.float32)
    spl_db = np.asarray(spl_db[valid], dtype=np.float32)
    times = np.asarray(times[valid], dtype=np.float32)

    orientation = "positive_depth"
    if float(np.nanmedian(distance)) < 0:
        distance = -distance
        orientation = "negated_depth"

    exponents = np.linspace(min_exponent, max_exponent, exponent_count)
    best_score = float("-inf")
    best_start = np.nan
    best_end = np.nan
    best_exponent = np.nan
    start = float(times[0])
    last = float(times[-1])
    while start + window_sec <= last + 1e-9:
        end = start + window_sec
        window = (times >= start) & (times <= end)
        if int(window.sum()) >= 3:
            window_distance = distance[window]
            observed = spl_db[window] - spl_db[window][0]
            initial_distance = float(window_distance[0])
            if np.isfinite(initial_distance) and initial_distance > 0 and np.nanmax(window_distance) > 0:
                distance_ratio = np.log10(
                    np.maximum(initial_distance / np.maximum(window_distance, 1e-6), 1e-6)
                )
                total_variance = float(np.sum((observed - np.mean(observed)) ** 2))
                if np.isfinite(total_variance) and total_variance > 0:
                    for exponent in exponents:
                        predicted = exponent * distance_ratio
                        residual = float(np.sum((observed - predicted) ** 2))
                        score = 1.0 - residual / total_variance
                        if np.isfinite(score) and score > best_score:
                            best_score = float(score)
                            best_start = start
                            best_end = end
                            best_exponent = float(exponent)
        start += slide_sec

    if best_score == float("-inf"):
        return np.nan, np.nan, np.nan, np.nan, orientation
    return best_score, best_start, best_end, best_exponent, orientation


def score_sample(sample_id: str, audio_root: Path, track_root: Path) -> dict:
    output = {
        SCORE_FIELD: np.nan,
        "sign_aware_window_start_sec": np.nan,
        "sign_aware_window_end_sec": np.nan,
        "sign_aware_best_exponent": np.nan,
        "sign_aware_depth_orientation": "",
        "sign_aware_missing_reason": "",
    }
    audio_path = audio_root / sample_id / "audio_features.npz"
    track_path = track_root / sample_id / "track.npz"
    if not audio_path.is_file() or not track_path.is_file():
        output["sign_aware_missing_reason"] = "missing_audio_or_track"
        return output
    try:
        audio = np.load(audio_path)
        track = np.load(track_path)
        audio_times = np.asarray(audio["frame_times_sec"], dtype=float)
        spl_db = np.asarray(audio["spl_curve_db"], dtype=float)
        track_times = np.asarray(track["frame_times_sec"], dtype=float)
        distance = np.asarray(track["source_depth_proxy_smooth"], dtype=float)
        common_times = np.linspace(
            max(audio_times.min(), track_times.min()),
            min(audio_times.max(), track_times.max()),
            min(len(audio_times), len(track_times)),
        )
        score, start, end, exponent, orientation = score_windows(
            np.interp(common_times, track_times, distance),
            np.interp(common_times, audio_times, spl_db),
            common_times,
        )
        output.update(
            {
                SCORE_FIELD: score,
                "sign_aware_window_start_sec": start,
                "sign_aware_window_end_sec": end,
                "sign_aware_best_exponent": exponent,
                "sign_aware_depth_orientation": orientation,
                "sign_aware_missing_reason": "" if np.isfinite(score) else "no_valid_window",
            }
        )
    except Exception as exc:
        output["sign_aware_missing_reason"] = f"processing_failed:{exc}"
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--track-root", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path)
    args = parser.parse_args()

    metrics = pd.read_csv(args.metrics_csv)
    rows = [
        score_sample(str(row.get("sample_id", "")), args.audio_root, args.track_root)
        for _, row in metrics.iterrows()
    ]
    scored = pd.concat([metrics.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.out_csv, index=False)

    values = pd.to_numeric(scored[SCORE_FIELD], errors="coerce")
    summary = {
        "row_count": int(len(scored)),
        "valid_count": int(values.notna().sum()),
        "mean_score": float(values.mean()) if values.notna().any() else None,
        "physical_score": float(100.0 * values.mean()) if values.notna().any() else None,
        "output": str(args.out_csv),
    }
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
