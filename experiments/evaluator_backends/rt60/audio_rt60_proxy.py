#!/usr/bin/env python3
"""Estimate sample-wise Schroeder apparent RT60 proxy from STARSS clap windows.

The default target is RT60-500. The implementation is center-frequency
parameterized so the same signal-processing path can be used for diagnostic
4000 Hz runs without maintaining a divergent copy.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from common import DEFAULT_OUTPUT_ROOT, ensure_dir, read_jsonl, write_csv, write_jsonl, write_text


CANDIDATE_SPECS = [
    ("T30_5_35", -5.0, -35.0, "diagnostic"),
    ("T20_5_25", -5.0, -25.0, "primary"),
    ("T10_5_15", -5.0, -15.0, "primary"),
    ("EDT_0_10", 0.0, -10.0, "primary"),
]
PRIMARY_ORDER = ["T20_5_25", "T10_5_15", "EDT_0_10"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    ap.add_argument("--manifest", default="")
    ap.add_argument("--max_samples", type=int, default=-1)
    ap.add_argument("--target_sr", type=int, default=24000)
    ap.add_argument("--center_hz", type=float, default=500.0)
    ap.add_argument("--input_format", choices=["auto", "foa", "mic", "mono"], default="auto")
    ap.add_argument("--min_peak_prom_db", type=float, default=10.0)
    ap.add_argument("--response_pre_peak_ms", type=float, default=5.0)
    ap.add_argument("--pre_noise_start_sec", type=float, default=0.50)
    ap.add_argument("--pre_noise_end_sec", type=float, default=0.05)
    ap.add_argument("--noise_cutoff_margin_db", type=float, default=10.0)
    ap.add_argument("--noise_cutoff_fallback_margin_db", type=float, default=6.0)
    ap.add_argument("--noise_frame_sec", type=float, default=0.010)
    ap.add_argument("--noise_hop_sec", type=float, default=0.002)
    ap.add_argument("--min_usable_decay_sec", type=float, default=0.050)
    ap.add_argument("--min_fit_support_sec", type=float, default=0.020)
    ap.add_argument("--min_r2", type=float, default=0.65)
    ap.add_argument("--min_proxy_sec", type=float, default=0.05)
    ap.add_argument("--max_proxy_sec", type=float, default=5.0)
    ap.add_argument("--disable_noise_compensation", action="store_true")
    ap.add_argument("--second_clap_min_delay_sec", type=float, default=0.250)
    ap.add_argument("--second_clap_rel_db", type=float, default=-8.0)
    ap.add_argument("--reject_suspected_second_clap", action="store_true")
    return ap.parse_args()


def metric_hz(center_hz: float) -> int:
    return int(round(center_hz))


def metric_field(center_hz: float, kind: str) -> str:
    return f"audio_{kind}_rt60_{metric_hz(center_hz)}_proxy"


def output_json_name(center_hz: float) -> str:
    return f"audio_rt60_{metric_hz(center_hz)}_proxy.jsonl"


def output_csv_name(center_hz: float) -> str:
    return f"audio_rt60_{metric_hz(center_hz)}_proxy.csv"


def output_summary_name(center_hz: float) -> str:
    return "summary_audio_rt60.md" if metric_hz(center_hz) == 500 else f"summary_audio_rt60_{metric_hz(center_hz)}.md"


def safe_db10(x: float | np.ndarray) -> float | np.ndarray:
    return 10.0 * np.log10(np.maximum(x, 1e-30))


def moving_average(x: np.ndarray, n: int) -> np.ndarray:
    n = max(1, int(n))
    if n <= 1:
        return x
    kernel = np.ones(n, dtype=np.float64) / float(n)
    return np.convolve(x, kernel, mode="same")


def linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    if len(x) < 3:
        return float("nan"), float("nan"), 0.0
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return float(slope), float(intercept), float(r2)


def octave_bandpass(y: np.ndarray, sr: int, center_hz: float) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt

    low = float(center_hz) / math.sqrt(2.0)
    high = float(center_hz) * math.sqrt(2.0)
    nyq = sr / 2.0
    if high >= nyq:
        high = nyq * 0.95
    if low <= 0 or low >= high:
        raise ValueError(f"invalid_octave_band:{low:.2f}-{high:.2f}@{sr}")
    sos = butter(4, [low / nyq, high / nyq], btype="bandpass", output="sos")
    return sosfiltfilt(sos, y).astype(np.float64)


def load_audio_multichannel(path: str, target_sr: int) -> tuple[np.ndarray, int, int, int]:
    import librosa

    y, sr = librosa.load(path, sr=None, mono=False)
    original_sr = int(sr)
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        y = y[None, :]
    if target_sr and target_sr > 0 and sr != target_sr:
        y = np.vstack([librosa.resample(ch, orig_sr=sr, target_sr=target_sr) for ch in y])
        sr = target_sr
    if y.shape[-1] == 0:
        raise ValueError("empty_audio")
    peak = float(np.max(np.abs(y)))
    if peak > 0:
        y = y / max(peak, 1e-12)
    return y.astype(np.float64), int(sr), original_sr, int(y.shape[0])


def detect_input_format(row: dict[str, Any], audio_path: str, channels: int, requested: str) -> str:
    if requested != "auto":
        return requested
    if channels == 1:
        return "mono"
    text = f"{row.get('audio_source', '')} {audio_path}".lower()
    parts = [p.lower() for p in Path(audio_path).parts]
    if "foa" in parts or "foa" in text:
        return "foa"
    if "mic" in parts or "mic" in text:
        return "mic"
    return "unknown_multi"


def channel_indices_for_format(detected_format: str, channels: int) -> list[int]:
    if detected_format == "foa":
        return [0]
    if detected_format == "mono":
        return [0]
    return list(range(channels))


def frame_power(signal: np.ndarray, sr: int, frame_sec: float, hop_sec: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = max(8, int(round(frame_sec * sr)))
    hop = max(1, int(round(hop_sec * sr)))
    starts = np.asarray([0], dtype=int) if len(signal) < frame else np.arange(0, len(signal) - frame + 1, hop, dtype=int)
    powers = np.asarray([float(np.mean(signal[s : s + frame] ** 2)) for s in starts], dtype=np.float64)
    powers = moving_average(powers, 5)
    centers = (starts + frame / 2.0) / float(sr)
    return starts, centers, powers


def detect_clap_peak(y: np.ndarray, sr: int, min_prom_db: float) -> tuple[int, float, float, bool]:
    smooth = moving_average(np.abs(y), max(1, int(round(0.003 * sr))))
    peak_idx = int(np.argmax(smooth))
    peak_amp = float(smooth[peak_idx])
    pre = smooth[: max(1, peak_idx - int(round(0.080 * sr)))]
    noise = float(np.percentile(pre, 75)) if len(pre) > int(0.020 * sr) else float(np.percentile(smooth, 20))
    prom_db = 20.0 * math.log10((peak_amp + 1e-12) / (noise + 1e-12))
    detected = bool(prom_db >= min_prom_db and peak_amp > 1e-4)
    return peak_idx, peak_amp, prom_db, detected


def estimate_pre_event_noise(
    band_full: np.ndarray,
    sr: int,
    peak_idx: int,
    pre_start_sec: float,
    pre_end_sec: float,
) -> tuple[float, float, str, int, int]:
    start = max(0, peak_idx - int(round(pre_start_sec * sr)))
    end = max(0, peak_idx - int(round(pre_end_sec * sr)))
    min_len = int(round(0.100 * sr))
    if end > start and (end - start) >= min_len:
        region = band_full[start:end]
        source = "pre_event_500ms_to_50ms"
        noise_start, noise_end = start, end
    else:
        start = int(round(0.80 * len(band_full)))
        end = len(band_full)
        region = band_full[start:end]
        source = "last_20_percent_fallback"
        noise_start, noise_end = start, end
    if len(region) == 0:
        region = band_full
        source = "full_signal_fallback"
        noise_start, noise_end = 0, len(band_full)
    noise_power = max(float(np.median(np.asarray(region, dtype=np.float64) ** 2)), 1e-30)
    return noise_power, float(safe_db10(noise_power)), source, noise_start, noise_end


def find_noise_cutoff(
    band_full: np.ndarray,
    sr: int,
    response_start_idx: int,
    noise_power: float,
    args: argparse.Namespace,
) -> tuple[int, float, str, float, np.ndarray, np.ndarray]:
    starts, centers, powers = frame_power(band_full, sr, args.noise_frame_sec, args.noise_hop_sec)
    frame_len = max(8, int(round(args.noise_frame_sec * sr)))
    min_samples = int(round(args.min_usable_decay_sec * sr))
    reasons = []
    for margin in (args.noise_cutoff_margin_db, args.noise_cutoff_fallback_margin_db):
        threshold = noise_power * (10.0 ** (float(margin) / 10.0))
        mask = (starts >= response_start_idx) & (powers > threshold)
        idxs = np.flatnonzero(mask)
        if len(idxs):
            last = int(idxs[-1])
            cutoff_idx = min(len(band_full), int(starts[last]) + frame_len)
            if cutoff_idx - response_start_idx >= min_samples:
                return cutoff_idx, float(margin), "last_reliable_frame", float(centers[last]), centers, powers
            reasons.append(f"margin_{margin:.1f}_too_short")
        else:
            reasons.append(f"margin_{margin:.1f}_no_frame")
    fallback = min(len(band_full), response_start_idx + max(min_samples, frame_len))
    return fallback, float(args.noise_cutoff_fallback_margin_db), ";".join(reasons), fallback / sr, centers, powers


def detect_suspected_second_clap(
    y: np.ndarray,
    sr: int,
    peak_idx: int,
    args: argparse.Namespace,
) -> tuple[bool, float | None, float | None]:
    from scipy.signal import find_peaks

    _starts, centers, powers = frame_power(y, sr, args.noise_frame_sec, args.noise_hop_sec)
    if len(powers) < 3:
        return False, None, None
    peak_time = peak_idx / sr
    main_mask = np.abs(centers - peak_time) <= 0.025
    main_power = float(np.max(powers[main_mask])) if int(np.sum(main_mask)) else float(np.max(powers))
    search_mask = centers >= peak_time + args.second_clap_min_delay_sec
    if int(np.sum(search_mask)) < 3:
        return False, None, None
    post = powers[search_mask]
    post_centers = centers[search_mask]
    threshold = main_power * (10.0 ** (args.second_clap_rel_db / 10.0))
    distance = max(1, int(round(0.050 / max(args.noise_hop_sec, 1e-6))))
    peaks, _ = find_peaks(post, distance=distance)
    strong = [int(idx) for idx in peaks if post[idx] >= threshold]
    if not strong:
        return False, None, None
    idx = strong[0]
    rel_db = 10.0 * math.log10((float(post[idx]) + 1e-30) / (main_power + 1e-30))
    return True, float(post_centers[idx]), float(rel_db)


def first_crossing(decay_db: np.ndarray, level_db: float, start: int = 0) -> int | None:
    if level_db == 0.0 and start == 0:
        return 0
    idxs = np.flatnonzero(decay_db[start:] <= level_db)
    return None if len(idxs) == 0 else int(start + idxs[0])


def empty_candidate(name: str, start_db: float, end_db: float, kind: str, reason: str) -> dict[str, Any]:
    return {
        "name": name,
        "start_db": start_db,
        "end_db": end_db,
        "kind": kind,
        "proxy": None,
        "slope": None,
        "intercept": None,
        "r2": None,
        "dynamic_range_db": None,
        "support_duration_sec": None,
        "fit_start_time_sec": None,
        "fit_end_time_sec": None,
        "fit_start_index": None,
        "fit_end_index": None,
        "valid": False,
        "invalid_reason": reason,
    }


def fit_decay_candidate(
    name: str,
    start_db: float,
    end_db: float,
    kind: str,
    times: np.ndarray,
    decay_db: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    start_idx = first_crossing(decay_db, start_db, 0)
    if start_idx is None:
        return empty_candidate(name, start_db, end_db, kind, "missing_start_crossing")
    end_idx = first_crossing(decay_db, end_db, start_idx + 1)
    if end_idx is None:
        return empty_candidate(name, start_db, end_db, kind, "missing_end_crossing")
    if end_idx <= start_idx:
        return empty_candidate(name, start_db, end_db, kind, "empty_crossing_region")
    tt = times[start_idx : end_idx + 1]
    dd = decay_db[start_idx : end_idx + 1]
    if len(tt) < 3:
        return empty_candidate(name, start_db, end_db, kind, "too_few_points")
    slope, intercept, r2 = linear_fit(tt, dd)
    support = float(tt[-1] - tt[0])
    dynamic = float(dd[0] - dd[-1])
    proxy = float(-60.0 / slope) if np.isfinite(slope) and slope < 0 else float("nan")
    reasons = []
    if not np.isfinite(slope) or slope >= 0:
        reasons.append("non_negative_or_invalid_slope")
    if r2 < args.min_r2:
        reasons.append(f"r2_below_{args.min_r2:.2f}")
    if support < args.min_fit_support_sec:
        reasons.append(f"support_below_{args.min_fit_support_sec:.3f}s")
    if not np.isfinite(proxy) or not (args.min_proxy_sec <= proxy <= args.max_proxy_sec):
        reasons.append("proxy_out_of_range")
    return {
        "name": name,
        "start_db": start_db,
        "end_db": end_db,
        "kind": kind,
        "proxy": proxy if np.isfinite(proxy) else None,
        "slope": float(slope) if np.isfinite(slope) else None,
        "intercept": float(intercept) if np.isfinite(intercept) else None,
        "r2": float(r2),
        "dynamic_range_db": dynamic,
        "support_duration_sec": support,
        "fit_start_time_sec": float(tt[0]),
        "fit_end_time_sec": float(tt[-1]),
        "fit_start_index": int(start_idx),
        "fit_end_index": int(end_idx),
        "valid": len(reasons) == 0,
        "invalid_reason": ";".join(reasons),
    }


def candidate_prefix(name: str) -> str:
    return name.lower()


def quality_for_summary(r2: float, dynamic: float, support: float, spread: float | None) -> str:
    if r2 >= 0.90 and dynamic >= 9.0 and support >= 0.05 and (spread is None or spread <= 0.25):
        return "high"
    if r2 >= 0.75 and dynamic >= 8.0:
        return "medium"
    return "low"


def estimate_channel(y: np.ndarray, sr: int, channel_index: int, args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {
        "channel_index": channel_index,
        "channel_valid": False,
        "selected_method": "invalid",
        "selected_proxy": None,
        "selected_r2": None,
        "selected_dynamic_range_db": None,
        "selected_support_duration_sec": None,
        "selected_fit_start_time_sec": None,
        "selected_fit_end_time_sec": None,
        "audio_peak_time_sec": None,
        "audio_peak_prom_db": None,
        "audio_event_detected": False,
        "pre_event_noise_power": None,
        "pre_event_noise_db": None,
        "noise_estimation_source": "",
        "noise_cutoff_time_sec": None,
        "cutoff_margin_db": None,
        "usable_decay_duration_sec": None,
        "usable_dynamic_range_db": None,
        "suspected_second_clap": False,
        "second_clap_time_sec": None,
        "second_clap_relative_db": None,
        "rejection_reason": "",
        "candidate_results": {},
        "_plot": {},
    }
    try:
        band_full = octave_bandpass(y, sr, args.center_hz)
        peak_idx, _peak_amp, peak_prom_db, peak_detected = detect_clap_peak(y, sr, args.min_peak_prom_db)
        peak_time = peak_idx / sr
        response_start_idx = max(0, peak_idx - int(round(args.response_pre_peak_ms * 0.001 * sr)))
        noise_power, noise_db, noise_source, noise_start, noise_end = estimate_pre_event_noise(
            band_full, sr, peak_idx, args.pre_noise_start_sec, args.pre_noise_end_sec
        )
        cutoff_idx, margin_db, cutoff_reason, _last_time, env_times, env_power = find_noise_cutoff(
            band_full, sr, response_start_idx, noise_power, args
        )
        suspected_second, second_time, second_rel_db = detect_suspected_second_clap(y, sr, peak_idx, args)
        result.update(
            {
                "audio_peak_time_sec": peak_time,
                "audio_peak_prom_db": peak_prom_db,
                "audio_event_detected": bool(peak_detected),
                "pre_event_noise_power": noise_power,
                "pre_event_noise_db": noise_db,
                "noise_estimation_source": noise_source,
                "noise_cutoff_time_sec": cutoff_idx / sr,
                "cutoff_margin_db": margin_db,
                "usable_decay_duration_sec": max(0.0, (cutoff_idx - response_start_idx) / sr),
                "suspected_second_clap": bool(suspected_second),
                "second_clap_time_sec": second_time,
                "second_clap_relative_db": second_rel_db,
            }
        )
        reasons = []
        if not peak_detected:
            reasons.append(f"weak_or_missing_clap_peak:{peak_prom_db:.2f}dB")
        if "last_reliable_frame" not in cutoff_reason:
            reasons.append(f"weak_noise_cutoff:{cutoff_reason}")
        if cutoff_idx - response_start_idx < int(round(args.min_usable_decay_sec * sr)):
            reasons.append("usable_decay_too_short")

        response = band_full[response_start_idx:cutoff_idx]
        squared = np.asarray(response, dtype=np.float64) ** 2
        if len(squared) < 8:
            raise ValueError("response_too_short")
        raw_edc = np.cumsum(squared[::-1])[::-1]
        if args.disable_noise_compensation:
            edc = raw_edc
            edc_mode = "raw"
        else:
            remaining = np.arange(len(squared), 0, -1, dtype=np.float64)
            corrected = raw_edc - noise_power * remaining
            positive = np.flatnonzero(corrected > 0)
            if len(positive) == 0 or positive[0] != 0:
                raise ValueError("noise_compensated_edc_nonpositive_at_start")
            first_nonpositive = np.flatnonzero(corrected <= 0)
            end = int(first_nonpositive[0]) if len(first_nonpositive) else len(corrected)
            edc = corrected[:end]
            edc_mode = "noise_compensated"
        if len(edc) < max(8, int(round(args.min_fit_support_sec * sr))):
            raise ValueError("edc_too_short_after_noise_compensation")

        times = (np.arange(len(edc), dtype=np.float64) + response_start_idx - peak_idx) / sr
        decay_db = np.asarray(safe_db10(edc / max(float(edc[0]), 1e-30)), dtype=np.float64)
        usable_dynamic = float(-np.nanmin(decay_db)) if len(decay_db) else 0.0
        result["usable_dynamic_range_db"] = usable_dynamic
        candidates = {
            name: fit_decay_candidate(name, start_db, end_db, kind, times, decay_db, args)
            for name, start_db, end_db, kind in CANDIDATE_SPECS
        }
        result["candidate_results"] = candidates
        selected = None
        for name in PRIMARY_ORDER:
            cand = candidates.get(name)
            if cand and cand.get("valid"):
                selected = cand
                break
        if selected is None:
            reasons.append(
                "no_valid_decay_candidate:"
                + json.dumps({name: candidates[name].get("invalid_reason") for name in PRIMARY_ORDER}, ensure_ascii=False)
            )
        if suspected_second and args.reject_suspected_second_clap:
            reasons.append("suspected_second_clap")
        if selected is not None:
            result.update(
                {
                    "selected_method": selected["name"],
                    "selected_proxy": selected["proxy"],
                    "selected_r2": selected["r2"],
                    "selected_dynamic_range_db": selected["dynamic_range_db"],
                    "selected_support_duration_sec": selected["support_duration_sec"],
                    "selected_fit_start_time_sec": selected["fit_start_time_sec"],
                    "selected_fit_end_time_sec": selected["fit_end_time_sec"],
                }
            )
        result["channel_valid"] = bool(selected is not None and not reasons)
        result["rejection_reason"] = ";".join(reasons)
        result["_plot"] = {
            "waveform": y,
            "band_full": band_full,
            "env_times": env_times,
            "env_power": env_power,
            "noise_start": noise_start,
            "noise_end": noise_end,
            "peak_idx": peak_idx,
            "response_start_idx": response_start_idx,
            "cutoff_idx": cutoff_idx,
            "times": times,
            "decay_db": decay_db,
            "edc_mode": edc_mode,
        }
    except Exception as exc:
        result["rejection_reason"] = str(exc)
    return result


def candidate_median(channel_results: list[dict[str, Any]], name: str) -> dict[str, Any]:
    valid = [
        ch["candidate_results"][name]
        for ch in channel_results
        if ch.get("channel_valid") and name in ch.get("candidate_results", {}) and ch["candidate_results"][name].get("valid")
    ]
    spec = next((s for s in CANDIDATE_SPECS if s[0] == name), (name, 0.0, 0.0, "aggregate"))
    if not valid:
        return empty_candidate(name, spec[1], spec[2], spec[3], "no_valid_channel")
    values = np.asarray([float(c["proxy"]) for c in valid], dtype=float)
    r2 = np.asarray([float(c["r2"]) for c in valid], dtype=float)
    dyn = np.asarray([float(c["dynamic_range_db"]) for c in valid], dtype=float)
    support = np.asarray([float(c["support_duration_sec"]) for c in valid], dtype=float)
    starts = np.asarray([float(c["fit_start_time_sec"]) for c in valid], dtype=float)
    ends = np.asarray([float(c["fit_end_time_sec"]) for c in valid], dtype=float)
    return {
        "name": name,
        "start_db": spec[1],
        "end_db": spec[2],
        "kind": spec[3],
        "proxy": float(np.median(values)),
        "r2": float(np.median(r2)),
        "dynamic_range_db": float(np.median(dyn)),
        "support_duration_sec": float(np.median(support)),
        "fit_start_time_sec": float(np.median(starts)),
        "fit_end_time_sec": float(np.median(ends)),
        "valid": True,
        "valid_channel_count": len(valid),
        "channel_values": values.tolist(),
        "invalid_reason": "",
    }


def aggregate_channels(channel_results: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate_candidates = {name: candidate_median(channel_results, name) for name, *_ in CANDIDATE_SPECS}
    selected_name = None
    selected = None
    for name in PRIMARY_ORDER:
        cand = aggregate_candidates.get(name)
        if cand and cand.get("valid"):
            selected_name = name
            selected = cand
            break
    if selected is None:
        return {
            "selected_method": "invalid",
            "selected": None,
            "aggregate_candidates": aggregate_candidates,
            "selected_channels": [],
            "valid_channel_count": 0,
            "channel_values": [],
            "channel_rt_std": None,
            "channel_rt_spread": None,
        }
    selected_channels = [
        ch["channel_index"]
        for ch in channel_results
        if ch.get("channel_valid")
        and selected_name in ch.get("candidate_results", {})
        and ch["candidate_results"][selected_name].get("valid")
    ]
    values = np.asarray(selected.get("channel_values", []), dtype=float)
    spread = float(np.max(values) - np.min(values)) if len(values) >= 2 else 0.0
    std = float(np.std(values)) if len(values) >= 2 else 0.0
    return {
        "selected_method": selected_name,
        "selected": selected,
        "aggregate_candidates": aggregate_candidates,
        "selected_channels": selected_channels,
        "valid_channel_count": len(values),
        "channel_values": values.tolist(),
        "channel_rt_std": std,
        "channel_rt_spread": spread,
    }


def make_base_record(sample_id: str, audio_path: str, outputs: Path, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "audio_path": audio_path,
        "center_hz": metric_hz(args.center_hz),
        "audio_event_detected": False,
        "audio_peak_time_sec": None,
        "audio_peak_prom_db": None,
        "tail_snr_db": None,
        "pre_event_noise_power": None,
        "pre_event_noise_db": None,
        "noise_estimation_source": "",
        "noise_cutoff_time_sec": None,
        "cutoff_margin_db": None,
        "usable_decay_duration_sec": None,
        "usable_dynamic_range_db": None,
        metric_field(args.center_hz, "early_decay"): "",
        metric_field(args.center_hz, "apparent"): "",
        "selected_method": "invalid",
        "rt_estimation_method": "invalid",
        "fit_slope_db_per_sec": "",
        "fit_r2": 0.0,
        "fit_dynamic_range_db": 0.0,
        "fit_support_duration_sec": 0.0,
        "fit_start_time_sec": "",
        "fit_end_time_sec": "",
        "audio_rt_valid": False,
        "audio_rt_quality": "invalid",
        "audio_rejection_reason": "",
        "analysis_sample_rate": args.target_sr,
        "original_sample_rate": "",
        "original_audio_channels": "",
        "input_format_arg": args.input_format,
        "detected_input_format": "",
        "selected_channel": "",
        "selected_channels": [],
        "valid_channel_count": 0,
        "channel_rt_values": [],
        "channel_rt_std": None,
        "channel_rt_spread": None,
        "channel_results": [],
        "suspected_second_clap": False,
        "second_clap_time_sec": None,
        "diagnostic_path": str(outputs / "audio_diagnostics" / f"{sample_id}.png"),
    }


def add_candidate_fields(record: dict[str, Any], aggregate_candidates: dict[str, dict[str, Any]]) -> None:
    for name, start_db, end_db, kind in CANDIDATE_SPECS:
        cand = aggregate_candidates.get(name, empty_candidate(name, start_db, end_db, kind, "missing"))
        prefix = candidate_prefix(name)
        record[f"{prefix}_proxy"] = cand.get("proxy")
        record[f"{prefix}_r2"] = cand.get("r2")
        record[f"{prefix}_dynamic_range_db"] = cand.get("dynamic_range_db")
        record[f"{prefix}_support_duration_sec"] = cand.get("support_duration_sec")
        record[f"{prefix}_fit_start_time_sec"] = cand.get("fit_start_time_sec")
        record[f"{prefix}_fit_end_time_sec"] = cand.get("fit_end_time_sec")
        record[f"{prefix}_valid"] = bool(cand.get("valid"))
        record[f"{prefix}_valid_channel_count"] = cand.get("valid_channel_count", 0)
        record[f"{prefix}_channel_values"] = cand.get("channel_values", [])
        record[f"{prefix}_invalid_reason"] = cand.get("invalid_reason")


def strip_plot_data(channel_result: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in channel_result.items() if k != "_plot"}


def estimate_one(row: dict[str, Any], outputs: Path, args: argparse.Namespace) -> dict[str, Any]:
    sample_id = row["sample_id"]
    audio_path = row.get("audio_path") or ""
    record = make_base_record(sample_id, audio_path, outputs, args)
    try:
        if not audio_path or not Path(audio_path).exists():
            raise ValueError("missing_audio_path")
        y_multi, sr, original_sr, original_channels = load_audio_multichannel(audio_path, args.target_sr)
        detected_format = detect_input_format(row, audio_path, original_channels, args.input_format)
        indices = channel_indices_for_format(detected_format, original_channels)
        record.update(
            {
                "analysis_sample_rate": sr,
                "original_sample_rate": original_sr,
                "original_audio_channels": original_channels,
                "detected_input_format": detected_format,
            }
        )
        channel_results = [estimate_channel(y_multi[idx], sr, idx, args) for idx in indices]
        aggregate = aggregate_channels(channel_results)
        add_candidate_fields(record, aggregate["aggregate_candidates"])
        selected = aggregate["selected"]
        first_for_meta = next((ch for ch in channel_results if ch.get("audio_peak_time_sec") is not None), None)
        if first_for_meta:
            second_times = [float(ch["second_clap_time_sec"]) for ch in channel_results if ch.get("second_clap_time_sec") is not None]
            record.update(
                {
                    "audio_event_detected": any(bool(ch.get("audio_event_detected")) for ch in channel_results),
                    "audio_peak_time_sec": first_for_meta.get("audio_peak_time_sec"),
                    "audio_peak_prom_db": first_for_meta.get("audio_peak_prom_db"),
                    "pre_event_noise_power": first_for_meta.get("pre_event_noise_power"),
                    "pre_event_noise_db": first_for_meta.get("pre_event_noise_db"),
                    "noise_estimation_source": first_for_meta.get("noise_estimation_source"),
                    "noise_cutoff_time_sec": first_for_meta.get("noise_cutoff_time_sec"),
                    "cutoff_margin_db": first_for_meta.get("cutoff_margin_db"),
                    "usable_decay_duration_sec": first_for_meta.get("usable_decay_duration_sec"),
                    "usable_dynamic_range_db": first_for_meta.get("usable_dynamic_range_db"),
                    "suspected_second_clap": any(bool(ch.get("suspected_second_clap")) for ch in channel_results),
                    "second_clap_time_sec": min(second_times) if second_times else None,
                }
            )
        record["selected_method"] = aggregate["selected_method"]
        record["rt_estimation_method"] = aggregate["selected_method"]
        record["selected_channels"] = aggregate["selected_channels"]
        record["selected_channel"] = aggregate["selected_channels"][0] if len(aggregate["selected_channels"]) == 1 else ""
        record["valid_channel_count"] = aggregate["valid_channel_count"]
        record["channel_rt_values"] = aggregate["channel_values"]
        record["channel_rt_std"] = aggregate["channel_rt_std"]
        record["channel_rt_spread"] = aggregate["channel_rt_spread"]
        if selected is not None:
            proxy = float(selected["proxy"])
            record[metric_field(args.center_hz, "early_decay")] = proxy
            record[metric_field(args.center_hz, "apparent")] = proxy
            record["fit_r2"] = float(selected["r2"])
            record["fit_dynamic_range_db"] = float(selected["dynamic_range_db"])
            record["fit_support_duration_sec"] = float(selected["support_duration_sec"])
            record["fit_start_time_sec"] = float(selected["fit_start_time_sec"])
            record["fit_end_time_sec"] = float(selected["fit_end_time_sec"])
            record["audio_rt_valid"] = True
            record["audio_rt_quality"] = quality_for_summary(
                float(selected["r2"]),
                float(selected["dynamic_range_db"]),
                float(selected["support_duration_sec"]),
                aggregate["channel_rt_spread"],
            )
            if aggregate["valid_channel_count"] < len(channel_results):
                record["audio_rejection_reason"] = "some_channels_invalid"
        else:
            reasons = [ch.get("rejection_reason", "") for ch in channel_results if ch.get("rejection_reason")]
            record["audio_rejection_reason"] = "no_valid_channel:" + " | ".join(reasons[:8])
        record["channel_results"] = [strip_plot_data(ch) for ch in channel_results]
        save_diagnostic(Path(record["diagnostic_path"]), y_multi, sr, channel_results, args)
    except Exception as exc:
        record["audio_rejection_reason"] = str(exc)
        try:
            if audio_path and Path(audio_path).exists():
                y_multi, sr, _original_sr, _channels = load_audio_multichannel(audio_path, args.target_sr)
                save_basic_diagnostic(Path(record["diagnostic_path"]), y_multi, sr, str(exc))
        except Exception:
            pass
    return record


def draw_fit(ax: Any, channel_result: dict[str, Any], candidate_name: str, color: str, label: str) -> None:
    plot = channel_result.get("_plot", {})
    cand = channel_result.get("candidate_results", {}).get(candidate_name)
    if not cand or not cand.get("valid") or not plot:
        return
    slope = cand.get("slope")
    intercept = cand.get("intercept")
    start = cand.get("fit_start_time_sec")
    end = cand.get("fit_end_time_sec")
    if slope is None or intercept is None or start is None or end is None:
        return
    xx = np.linspace(float(start), float(end), 32)
    yy = float(slope) * xx + float(intercept)
    ax.plot(xx, yy, color=color, linewidth=2.0, label=label)
    ax.axvspan(float(start), float(end), color=color, alpha=0.08)


def save_diagnostic(path: Path, y_multi: np.ndarray, sr: int, channel_results: list[dict[str, Any]], args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_dir(path.parent)
    rows = max(1, len(channel_results))
    fig, axes = plt.subplots(rows, 3, figsize=(16, max(3.2, 3.0 * rows)), squeeze=False)
    for row_idx, ch in enumerate(channel_results):
        plot = ch.get("_plot", {})
        ch_idx = int(ch.get("channel_index", row_idx))
        y = y_multi[ch_idx]
        t = np.arange(len(y)) / sr
        ax0, ax1, ax2 = axes[row_idx]
        ax0.plot(t, y, linewidth=0.45)
        peak_time = ch.get("audio_peak_time_sec")
        if peak_time is not None:
            ax0.axvline(float(peak_time), color="r", linestyle="--", label="clap peak")
        if plot:
            ns = plot.get("noise_start")
            ne = plot.get("noise_end")
            if ns is not None and ne is not None:
                ax0.axvspan(float(ns) / sr, float(ne) / sr, color="gray", alpha=0.18, label="pre-event noise")
        if ch.get("second_clap_time_sec") is not None:
            ax0.axvline(float(ch["second_clap_time_sec"]), color="m", linestyle=":", label="suspected second clap")
        ax0.set_title(f"ch{ch_idx} waveform")
        ax0.set_xlabel("time (s)")
        ax0.legend(fontsize=7, loc="upper right")

        if plot:
            band = plot.get("band_full")
            if band is not None:
                ax1.plot(t, band, linewidth=0.45)
            if peak_time is not None:
                ax1.axvline(float(peak_time), color="r", linestyle="--", label="clap peak")
            cutoff_idx = plot.get("cutoff_idx")
            if cutoff_idx is not None:
                ax1.axvline(float(cutoff_idx) / sr, color="orange", linestyle="--", label="noise cutoff")
            response_start = plot.get("response_start_idx")
            if response_start is not None:
                ax1.axvline(float(response_start) / sr, color="green", linestyle=":", label="response start")
            ax1.set_title(f"{int(round(args.center_hz))}Hz bandpassed")
            ax1.set_xlabel("time (s)")
            ax1.legend(fontsize=7, loc="upper right")

            times = plot.get("times")
            decay_db = plot.get("decay_db")
            if times is not None and decay_db is not None and len(times):
                ax2.plot(times, decay_db, linewidth=1.0, label="sample-wise Schroeder EDC")
                for level in (-5, -10, -15, -25, -35):
                    ax2.axhline(level, color="gray", linewidth=0.5, linestyle=":")
                selected = ch.get("selected_method")
                if selected != "invalid":
                    draw_fit(ax2, ch, selected, "red", f"selected {selected}")
                draw_fit(ax2, ch, "T30_5_35", "tab:purple", "T30 diagnostic")
                ax2.set_ylim(bottom=max(float(np.nanmin(decay_db)) - 5.0, -80.0), top=3.0)
        ax2.set_title(f"ch{ch_idx} EDC")
        ax2.set_xlabel("time after peak (s)")
        ax2.set_ylabel("dB")
        ax2.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_basic_diagnostic(path: Path, y_multi: np.ndarray, sr: int, reason: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_dir(path.parent)
    t = np.arange(y_multi.shape[-1]) / sr
    fig, ax = plt.subplots(1, 1, figsize=(11, 3.5))
    for idx, y in enumerate(y_multi[:4]):
        ax.plot(t, y, linewidth=0.45, label=f"ch{idx}")
    ax.set_title(f"Audio diagnostic failed: {reason[:160]}")
    ax.set_xlabel("time (s)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def field_order(args: argparse.Namespace) -> list[str]:
    fields = [
        "sample_id",
        "audio_path",
        "center_hz",
        "audio_event_detected",
        "audio_peak_time_sec",
        "audio_peak_prom_db",
        "pre_event_noise_power",
        "pre_event_noise_db",
        "noise_estimation_source",
        "noise_cutoff_time_sec",
        "cutoff_margin_db",
        "usable_decay_duration_sec",
        "usable_dynamic_range_db",
        metric_field(args.center_hz, "early_decay"),
        metric_field(args.center_hz, "apparent"),
        "selected_method",
        "rt_estimation_method",
        "fit_r2",
        "fit_dynamic_range_db",
        "fit_support_duration_sec",
        "fit_start_time_sec",
        "fit_end_time_sec",
        "audio_rt_valid",
        "audio_rt_quality",
        "audio_rejection_reason",
        "analysis_sample_rate",
        "original_sample_rate",
        "original_audio_channels",
        "input_format_arg",
        "detected_input_format",
        "selected_channel",
        "selected_channels",
        "valid_channel_count",
        "channel_rt_values",
        "channel_rt_std",
        "channel_rt_spread",
        "suspected_second_clap",
        "second_clap_time_sec",
        "diagnostic_path",
        "channel_results",
    ]
    for name, *_ in CANDIDATE_SPECS:
        prefix = candidate_prefix(name)
        fields.extend(
            [
                f"{prefix}_proxy",
                f"{prefix}_r2",
                f"{prefix}_dynamic_range_db",
                f"{prefix}_support_duration_sec",
                f"{prefix}_fit_start_time_sec",
                f"{prefix}_fit_end_time_sec",
                f"{prefix}_valid",
                f"{prefix}_valid_channel_count",
                f"{prefix}_channel_values",
                f"{prefix}_invalid_reason",
            ]
        )
    return fields


def summarize_records(records: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    proxy_field = metric_field(args.center_hz, "apparent")
    valid = [r for r in records if r.get("audio_rt_valid")]
    vals = np.asarray([float(r[proxy_field]) for r in valid], dtype=float) if valid else np.asarray([])
    failure_counter: Counter[str] = Counter()
    for r in records:
        if r.get("audio_rt_valid"):
            continue
        reason = str(r.get("audio_rejection_reason", ""))
        if "weak_or_missing_clap_peak" in reason:
            failure_counter["weak_or_missing_clap_peak"] += 1
        if "no_valid_decay_candidate" in reason:
            failure_counter["no_valid_decay_candidate"] += 1
        if "noise_compensated_edc" in reason or "edc_too_short" in reason:
            failure_counter["noise_compensation_or_edc_too_short"] += 1
        if "weak_noise_cutoff" in reason:
            failure_counter["weak_noise_cutoff"] += 1
        if "suspected_second_clap" in reason:
            failure_counter["suspected_second_clap"] += 1
        if not reason:
            failure_counter["unknown"] += 1
    return {
        "valid": valid,
        "vals": vals,
        "method_counts": Counter(r.get("selected_method", "invalid") for r in records),
        "quality_counts": Counter(r.get("audio_rt_quality", "invalid") for r in records),
        "format_counts": Counter(r.get("detected_input_format", "") for r in records),
        "failure_counts": failure_counter,
        "suspected_second_clap_count": sum(bool(r.get("suspected_second_clap")) for r in records),
        "multi_channel_median_count": sum(int(r.get("valid_channel_count") or 0) >= 2 for r in records),
    }


def write_summary(outputs: Path, records: list[dict[str, Any]], args: argparse.Namespace) -> None:
    mhz = metric_hz(args.center_hz)
    proxy_field = metric_field(args.center_hz, "apparent")
    stats = summarize_records(records, args)
    vals = stats["vals"]
    valid = stats["valid"]
    lines = [
        f"# Audio Sample-Wise Schroeder Apparent RT60-{mhz} Proxy Summary",
        "",
        f"Primary output field: `{proxy_field}`.",
        "This is an apparent clap-response proxy, not a standard sweep/RIR RT60.",
        "Schroeder integration is performed only over the response segment ending at the detected noise cutoff.",
        "",
        f"- sample_count: {len(records)}",
        f"- valid_audio_proxy_count: {len(valid)}",
        f"- valid_audio_proxy_rate: {(len(valid) / len(records) if records else 0):.3f}",
        f"- median_proxy: {(float(np.median(vals)) if len(vals) else 0):.4f}",
        f"- mean_proxy: {(float(np.mean(vals)) if len(vals) else 0):.4f}",
        f"- selected_method_counts: {dict(stats['method_counts'])}",
        f"- audio_rt_quality_counts: {dict(stats['quality_counts'])}",
        f"- detected_input_format_counts: {dict(stats['format_counts'])}",
        f"- failure_reason_counts: {dict(stats['failure_counts'])}",
        f"- suspected_second_clap_count: {stats['suspected_second_clap_count']}",
        f"- multi_channel_median_count: {stats['multi_channel_median_count']}",
        f"- input_format_arg: {args.input_format}",
        f"- center_hz: {args.center_hz}",
        f"- octave_band_hz: {args.center_hz / math.sqrt(2.0):.1f}-{args.center_hz * math.sqrt(2.0):.1f}",
        f"- min_fit_support_sec: {args.min_fit_support_sec}",
        f"- proxy_range_sec: {args.min_proxy_sec}-{args.max_proxy_sec}",
        f"- noise_cutoff_margin_db: {args.noise_cutoff_margin_db}",
        f"- noise_cutoff_fallback_margin_db: {args.noise_cutoff_fallback_margin_db}",
        f"- diagnostics_dir: `{outputs / 'audio_diagnostics'}`",
        "",
        "Method selection order: T20_5_25 -> T10_5_15 -> EDT_0_10. T30_5_35 is diagnostic.",
        "",
        "Valid examples:",
    ]
    for r in valid[:20]:
        lines.append(
            f"- `{r['sample_id']}`: {r.get(proxy_field)} sec, method={r.get('selected_method')}, "
            f"channels={r.get('selected_channels')}, r2={r.get('fit_r2')}"
        )
    if len(records) != len(valid):
        lines.append("")
        lines.append("Invalid examples:")
        count = 0
        for r in records:
            if not r.get("audio_rt_valid"):
                lines.append(f"- `{r['sample_id']}`: {r.get('audio_rejection_reason', '')}")
                count += 1
                if count >= 30:
                    lines.append("- ...")
                    break
    write_text(outputs / output_summary_name(args.center_hz), "\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    outputs = ensure_dir(output_root / "outputs")
    manifest = Path(args.manifest) if args.manifest else outputs / "manifest_starss_claps.jsonl"
    rows = read_jsonl(manifest)
    if args.max_samples and args.max_samples > 0:
        rows = rows[: args.max_samples]
    records = [estimate_one(row, outputs, args) for row in rows]
    write_jsonl(outputs / output_json_name(args.center_hz), records)
    write_csv(outputs / output_csv_name(args.center_hz), records, field_order(args))
    write_summary(outputs, records, args)
    print(
        {
            "center_hz": metric_hz(args.center_hz),
            "sample_count": len(records),
            "valid": sum(bool(r.get("audio_rt_valid")) for r in records),
            "selected_methods": dict(Counter(r.get("selected_method") for r in records)),
        }
    )


if __name__ == "__main__":
    main()
