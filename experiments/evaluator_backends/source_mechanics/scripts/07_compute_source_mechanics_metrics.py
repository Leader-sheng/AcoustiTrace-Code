from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from common import ensure_dir, load_config, read_csv_dicts, safe_float, write_csv_dicts, run_cmd, tool_cmd


def _parse_pgm_stream(data: bytes) -> list[np.ndarray]:
    frames = []
    pos = 0
    n = len(data)

    def read_token() -> str | None:
        nonlocal pos
        while pos < n:
            b = data[pos]
            if b in b" \t\r\n":
                pos += 1
                continue
            if b == ord("#"):
                while pos < n and data[pos] not in b"\r\n":
                    pos += 1
                continue
            break
        if pos >= n:
            return None
        start = pos
        while pos < n and data[pos] not in b" \t\r\n":
            pos += 1
        return data[start:pos].decode("ascii", errors="ignore")

    while pos < n:
        magic = read_token()
        if magic is None:
            break
        if magic != "P5":
            break
        w_s = read_token()
        h_s = read_token()
        max_s = read_token()
        if not w_s or not h_s or not max_s:
            break
        try:
            width = int(w_s)
            height = int(h_s)
            maxval = int(max_s)
        except Exception:
            break
        if maxval > 255:
            break
        while pos < n and data[pos] in b" \t\r\n":
            pos += 1
        frame_bytes = width * height
        if pos + frame_bytes > n:
            break
        frame = np.frombuffer(data[pos : pos + frame_bytes], dtype=np.uint8).reshape((height, width)).astype(np.float32)
        frames.append(frame)
        pos += frame_bytes
    return frames


def read_gray_frames_ffmpeg(video_path: Path, start_t: float, end_t: float, fps_sample: int) -> list[np.ndarray]:
    duration = max(0.05, end_t - start_t)
    vf = f"fps={fps_sample},scale=160:-1,format=gray"
    cmd = tool_cmd("ffmpeg") + [
        "-v",
        "error",
        "-ss",
        f"{start_t:.6f}",
        "-t",
        f"{duration:.6f}",
        "-i",
        str(video_path),
        "-vf",
        vf,
        "-f",
        "image2pipe",
        "-vcodec",
        "pgm",
        "pipe:1",
    ]
    cp = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if cp.returncode != 0 or not cp.stdout:
        return []
    return _parse_pgm_stream(cp.stdout)


def load_audio_file(audio_path: Path, sr: int) -> tuple[np.ndarray, int]:
    import librosa

    y, sr_loaded = librosa.load(str(audio_path), sr=sr, mono=True)
    return y.astype(np.float32), sr_loaded


def extract_audio_cache(video_row: dict, audio_dir: Path, sr: int) -> Path | None:
    audio_path = Path(video_row.get("audio_path") or "")
    video_path = Path(video_row["video_path"])
    cache = audio_dir / f"{video_row['video_id']}.wav"
    if cache.exists():
        return cache
    src = audio_path if audio_path.exists() else video_path
    if not src.exists():
        return None
    ensure_dir(audio_dir)
    cmd = tool_cmd("ffmpeg") + [
        "-y",
        "-i",
        str(src),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sr),
        str(cache),
    ]
    cp = run_cmd(cmd, check=False)
    if cp.returncode != 0 or not cache.exists():
        return None
    return cache


def motion_proxy(video_path: Path, peak_sec: float, fps_sample: int, before: float, after: float) -> dict:
    try:
        import cv2
    except Exception:
        cv2 = None

    start_t = max(0.0, peak_sec - before)
    end_t = max(start_t + 0.05, peak_sec + after)
    times = np.arange(start_t, end_t + 1e-6, 1.0 / fps_sample)
    frames = []

    try:
        if cv2 is not None:
            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                for t in times:
                    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        continue
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
                    frames.append(gray)
                cap.release()
        if len(frames) < 2:
            try:
                from moviepy.editor import VideoFileClip

                clip = VideoFileClip(str(video_path))
                for t in times:
                    if t < 0 or t > (clip.duration or 0):
                        continue
                    frame = clip.get_frame(float(t))
                    gray = frame.mean(axis=2).astype(np.float32)
                    frames.append(gray)
                clip.close()
            except Exception:
                frames = []
        if len(frames) < 2:
            frames = read_gray_frames_ffmpeg(video_path, start_t, end_t, fps_sample)
    except Exception:
        return {
            "visual_motion_mean": float("nan"),
            "visual_motion_max": float("nan"),
            "visual_motion_area_ratio": float("nan"),
            "visual_action_strength": float("nan"),
            "motion_failed": True,
        }

    if len(frames) < 2:
        return {
            "visual_motion_mean": float("nan"),
            "visual_motion_max": float("nan"),
            "visual_motion_area_ratio": float("nan"),
            "visual_action_strength": float("nan"),
            "motion_failed": True,
        }

    diffs = []
    ratios = []
    for a, b in zip(frames[:-1], frames[1:]):
        d = np.abs(b - a)
        diffs.append(float(np.mean(d)))
        ratios.append(float(np.mean(d > 12.0)))
    return {
        "visual_motion_mean": float(np.mean(diffs)),
        "visual_motion_max": float(np.max(diffs)),
        "visual_motion_area_ratio": float(np.mean(ratios)),
        "visual_action_strength": float(np.max(diffs)),
        "motion_failed": False,
    }


def spectral_feats(y: np.ndarray, sr: int) -> dict:
    import librosa

    if len(y) < sr // 20:
        return {
            "spectral_flatness": float("nan"),
            "spectral_centroid": float("nan"),
            "spectral_bandwidth": float("nan"),
            "spectral_rolloff": float("nan"),
            "hf_ratio": float("nan"),
        }
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=128)) ** 2
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr).mean()
    flatness = librosa.feature.spectral_flatness(S=S).mean()
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr).mean()
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr).mean()
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
    power = S.mean(axis=1)
    low = power[(freqs >= 200) & (freqs <= 1000)].sum()
    high = power[(freqs >= 3000) & (freqs <= 8000)].sum()
    hf_ratio = float(high / (low + high + 1e-8))
    return {
        "spectral_flatness": float(flatness),
        "spectral_centroid": float(centroid),
        "spectral_bandwidth": float(bandwidth),
        "spectral_rolloff": float(rolloff),
        "hf_ratio": hf_ratio,
    }


def envelope_metrics(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    import librosa
    from scipy.ndimage import gaussian_filter1d

    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=128)[0]
    env = gaussian_filter1d(rms, sigma=1.0)
    times = librosa.frames_to_time(np.arange(len(env)), sr=sr, hop_length=128)
    return env, times


def robust_envelope_noise_floor(env: np.ndarray, peak_idx: int) -> float:
    """Estimate a robust event-local noise floor from an RMS envelope."""
    finite = np.asarray(env, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return 1e-8
    pre_peak = np.asarray(env[: max(0, peak_idx)], dtype=float)
    pre_peak = pre_peak[np.isfinite(pre_peak)]
    candidates = [float(np.percentile(finite, 10))]
    if len(pre_peak) >= 3:
        candidates.append(float(np.percentile(pre_peak, 20)))
    floor = float(np.median(candidates))
    return max(floor, 1e-8)


def decay_floor_diagnostics(env: np.ndarray, peak_idx: int) -> dict:
    floor = robust_envelope_noise_floor(env, peak_idx)
    peak = float(np.nanmax(env)) if len(env) else float("nan")
    if not np.isfinite(peak) or peak <= 0:
        peak_to_floor_db = float("nan")
    else:
        peak_to_floor_db = float(20.0 * np.log10(max(peak, 1e-8) / max(floor, 1e-8)))
    return {
        "noise_floor_rms": float(floor),
        "peak_envelope_rms": float(peak),
        "peak_to_floor_db": peak_to_floor_db,
    }


def fit_decay(env: np.ndarray, times: np.ndarray, peak_idx: int, start_after: float, end_after: float) -> dict:
    from scipy.optimize import curve_fit

    floor_diag = decay_floor_diagnostics(env, peak_idx)

    if len(env) < 5 or peak_idx >= len(env) - 2:
        return {
            "decay_r2": float("nan"),
            "decay_lambda": float("nan"),
            "decay_fit_failed": True,
            "decay_fit_num_points": 0,
            "decay_r2_clipped": float("nan"),
            "envelope_residual_std": float("nan"),
            "envelope_residual_mae": float("nan"),
            "tail_residual_mae_db": float("nan"),
            "max_drop_db": float("nan"),
            "truncation_anomaly_score": float("nan"),
            "fit_support_ratio": float("nan"),
            "decay_shape_score": float("nan"),
            **floor_diag,
        }

    peak_t = times[peak_idx]
    start_t = peak_t + start_after
    end_t = min(times[-1], peak_t + end_after)
    mask = (times >= start_t) & (times <= end_t)
    x = times[mask] - peak_t
    y = env[mask]
    fit_support_ratio = float(max(0.0, end_t - start_t) / max(end_after - start_after, 1e-8))
    fit_support_ratio = float(max(0.0, min(1.0, fit_support_ratio)))
    if len(x) < 5:
        return {
            "decay_r2": float("nan"),
            "decay_lambda": float("nan"),
            "decay_fit_failed": True,
            "decay_fit_num_points": len(x),
            "decay_r2_clipped": float("nan"),
            "envelope_residual_std": float("nan"),
            "envelope_residual_mae": float("nan"),
            "tail_residual_mae_db": float("nan"),
            "max_drop_db": float("nan"),
            "truncation_anomaly_score": float("nan"),
            "fit_support_ratio": fit_support_ratio,
            "decay_shape_score": float("nan"),
            **floor_diag,
        }

    floor = max(np.percentile(y, 5), 1e-6)

    def fn(t, A, lam, C):
        return A * np.exp(-lam * t) + C

    A0 = max(float(y[0] - floor), 1e-3)
    lam0 = 5.0
    C0 = floor
    try:
        popt, _ = curve_fit(fn, x, y, p0=[A0, lam0, C0], maxfev=10000)
        fit = fn(x, *popt)
        ss_res = float(np.sum((y - fit) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2)) + 1e-8
        r2 = 1.0 - ss_res / ss_tot
        residual = y - fit
        log_env = 20 * np.log10(np.maximum(y, 1e-6))
        fit_log = 20 * np.log10(np.maximum(fit, 1e-6))
        residual_db = log_env - fit_log
        tail_residual_mae_db = float(np.mean(np.abs(residual_db)))
        drop = np.diff(log_env)
        max_drop_db = float(np.max(-drop)) if len(drop) else float("nan")
        r2_clipped = float(max(min(r2, 1.0), 0.0))
        peak_to_floor_db = safe_float(floor_diag.get("peak_to_floor_db"))
        denom_db = max(peak_to_floor_db, 1e-6) if np.isfinite(peak_to_floor_db) else 1e-6
        decay_shape_score = float(r2_clipped * np.exp(-tail_residual_mae_db / denom_db))
        return {
            "decay_r2": float(r2),
            "decay_lambda": float(popt[1]),
            "decay_fit_failed": False,
            "decay_fit_num_points": int(len(x)),
            "decay_r2_clipped": float(max(min(r2, 1.0), -1.0)),
            "envelope_residual_std": float(np.std(residual_db)),
            "envelope_residual_mae": tail_residual_mae_db,
            "tail_residual_mae_db": tail_residual_mae_db,
            "max_drop_db": max_drop_db,
            "truncation_anomaly_score": max_drop_db,
            "fit_support_ratio": fit_support_ratio,
            "decay_shape_score": decay_shape_score,
            **floor_diag,
        }
    except Exception:
        log_env = 20 * np.log10(np.maximum(y, 1e-6))
        drop = np.diff(log_env)
        max_drop_db = float(np.max(-drop)) if len(drop) else float("nan")
        return {
            "decay_r2": float("nan"),
            "decay_lambda": float("nan"),
            "decay_fit_failed": True,
            "decay_fit_num_points": int(len(x)),
            "decay_r2_clipped": float("nan"),
            "envelope_residual_std": float("nan"),
            "envelope_residual_mae": float("nan"),
            "tail_residual_mae_db": float("nan"),
            "max_drop_db": max_drop_db,
            "truncation_anomaly_score": max_drop_db,
            "fit_support_ratio": fit_support_ratio,
            "decay_shape_score": float("nan"),
            **floor_diag,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_root = Path(cfg["output"]["root"])
    matched_rows = read_csv_dicts(out_root / "matched_events" / "matched_av_events.csv")
    videos = read_csv_dicts(out_root / "index" / "videos.csv")
    video_map = {r["video_id"]: r for r in videos}
    sr = int(cfg["audio"]["sample_rate"])
    before = float(cfg["visual"]["motion_window_before_sec"])
    after = float(cfg["visual"]["motion_window_after_sec"])
    fps_sample = int(cfg["visual"]["fps_sample"])
    audio_dir = ensure_dir(out_root / "metrics" / "audio_cache")

    event_rows = []
    by_video = defaultdict(list)

    for row in matched_rows:
        vid = row["video_id"]
        vrow = video_map.get(vid)
        if not vrow:
            continue
        audio_cache = extract_audio_cache(vrow, audio_dir, sr)
        if audio_cache is None:
            event_rows.append({**row, "failed_reason": "audio_extract_failed"})
            continue
        try:
            y, _ = load_audio_file(audio_cache, sr)
        except Exception as e:
            event_rows.append({**row, "failed_reason": f"audio_load_failed:{e}"})
            continue

        audio_peak = safe_float(row["audio_peak_sec"])
        audio_start = max(0.0, audio_peak - float(cfg["audio"]["event_pre_sec"]))
        audio_end = min(len(y) / sr, audio_peak + float(cfg["audio"]["event_post_sec"]))
        s0 = int(audio_start * sr)
        s1 = max(s0 + 1, int(audio_end * sr))
        seg = y[s0:s1]
        if len(seg) < 64:
            event_rows.append({**row, "failed_reason": "audio_segment_too_short"})
            continue

        env, times = envelope_metrics(seg, sr)
        peak_idx = int(np.argmax(env))
        spec = spectral_feats(seg, sr)
        decay = fit_decay(
            env,
            times,
            peak_idx,
            float(cfg["audio"]["decay_fit_start_after_peak_sec"]),
            float(cfg["audio"]["decay_fit_end_after_peak_sec"]),
        )
        motion = motion_proxy(
            Path(vrow["video_path"]),
            safe_float(row["visual_peak_sec"]),
            fps_sample=fps_sample,
            before=before,
            after=after,
        )

        event_rms = float(np.sqrt(np.mean(seg**2)))
        event_rms_db = float(20.0 * np.log10(event_rms + 1e-8))

        out = {
            **row,
            "event_rms": event_rms,
            "event_rms_db": event_rms_db,
            "peak_amplitude": float(np.max(np.abs(seg))),
            **motion,
            **spec,
            **decay,
        }
        event_rows.append(out)
        by_video[vid].append(out)

    # per-video aggregation
    video_rows = []
    for vid, rows in by_video.items():
        vis = np.array([safe_float(r.get("visual_action_strength")) for r in rows], dtype=float)
        aud = np.array([safe_float(r.get("event_rms_db")) for r in rows], dtype=float)
        valid = np.isfinite(vis) & np.isfinite(aud)
        spearman = float("nan")
        pval = float("nan")
        pairwise = float("nan")
        npairs = 0
        if valid.sum() >= 3 and np.unique(vis[valid]).size >= 2 and np.unique(aud[valid]).size >= 2:
            try:
                from scipy.stats import spearmanr

                spearman, pval = spearmanr(vis[valid], aud[valid])
            except Exception:
                spearman, pval = float("nan"), float("nan")
            correct = 0
            total = 0
            idxs = np.where(valid)[0]
            for i in range(len(idxs)):
                for j in range(i + 1, len(idxs)):
                    ii, jj = idxs[i], idxs[j]
                    if vis[ii] == vis[jj] or aud[ii] == aud[jj]:
                        continue
                    total += 1
                    if (vis[ii] > vis[jj] and aud[ii] > aud[jj]) or (vis[ii] < vis[jj] and aud[ii] < aud[jj]):
                        correct += 1
            if total > 0:
                pairwise = correct / total
                npairs = total
        video_rows.append(
            {
                "video_id": vid,
                "video_path": rows[0]["video_path"],
                "num_matched_events": len(rows),
                "mean_visual_action_strength": float(np.nanmean(vis)) if len(vis) else float("nan"),
                "mean_event_rms_db": float(np.nanmean(aud)) if len(aud) else float("nan"),
                "spearman_motion_rms": spearman,
                "spearman_pvalue": pval,
                "pairwise_force_loudness_acc": pairwise,
                "num_pairs": npairs,
                "mean_decay_r2": float(np.nanmean([safe_float(r.get("decay_r2")) for r in rows])) if rows else float("nan"),
                "mean_decay_shape_score": float(np.nanmean([safe_float(r.get("decay_shape_score")) for r in rows])) if rows else float("nan"),
                "mean_decay_lambda": float(np.nanmean([safe_float(r.get("decay_lambda")) for r in rows])) if rows else float("nan"),
                "mean_max_drop_db": float(np.nanmean([safe_float(r.get("max_drop_db")) for r in rows])) if rows else float("nan"),
                "mean_peak_to_floor_db": float(np.nanmean([safe_float(r.get("peak_to_floor_db")) for r in rows])) if rows else float("nan"),
                "mean_tail_residual_mae_db": float(np.nanmean([safe_float(r.get("tail_residual_mae_db")) for r in rows])) if rows else float("nan"),
                "mean_fit_support_ratio": float(np.nanmean([safe_float(r.get("fit_support_ratio")) for r in rows])) if rows else float("nan"),
                "mean_spectral_flatness": float(np.nanmean([safe_float(r.get("spectral_flatness")) for r in rows])) if rows else float("nan"),
                "mean_spectral_centroid": float(np.nanmean([safe_float(r.get("spectral_centroid")) for r in rows])) if rows else float("nan"),
                "mean_hf_ratio": float(np.nanmean([safe_float(r.get("hf_ratio")) for r in rows])) if rows else float("nan"),
                "num_valid_events": int(sum(np.isfinite(vis) & np.isfinite(aud))),
            }
        )

    metrics_dir = ensure_dir(out_root / "metrics")
    def union_keys(rows):
        keys = []
        seen = set()
        for row in rows:
            for k in row.keys():
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
        return keys

    event_fieldnames = union_keys(event_rows)
    video_fieldnames = union_keys(video_rows)
    if event_rows:
        write_csv_dicts(metrics_dir / "source_mechanics_event_metrics.csv", event_rows, event_fieldnames)
    else:
        write_csv_dicts(metrics_dir / "source_mechanics_event_metrics.csv", [], [])
    if video_rows:
        write_csv_dicts(metrics_dir / "source_mechanics_video_metrics.csv", video_rows, video_fieldnames)
    else:
        write_csv_dicts(metrics_dir / "source_mechanics_video_metrics.csv", [], [])

    def safe_mean(values):
        arr = np.array([v for v in values if np.isfinite(v)], dtype=float)
        return float(arr.mean()) if len(arr) else float("nan")

    summary = {
        "num_videos": len(videos),
        "num_videos_with_audio": sum(1 for r in videos if str(r.get("has_audio")).lower() in {"1", "true", "yes"}),
        "num_visual_events": len(read_csv_dicts(out_root / "ov_avel" / "visual_events.csv")),
        "num_audio_events": len(read_csv_dicts(out_root / "flexsed" / "audio_events.csv")),
        "num_matched_events": len(matched_rows),
        "match_rate": len(matched_rows) / max(1, min(len(read_csv_dicts(out_root / "ov_avel" / "visual_events.csv")), len(read_csv_dicts(out_root / "flexsed" / "audio_events.csv")))),
        "num_videos_with_3plus_events": sum(1 for r in video_rows if r["num_matched_events"] >= 3),
        "mean_spearman_motion_rms": safe_mean([safe_float(r.get("spearman_motion_rms")) for r in video_rows]),
        "median_spearman_motion_rms": float(np.nanmedian([safe_float(r.get("spearman_motion_rms")) for r in video_rows])) if video_rows else float("nan"),
        "num_valid_videos_for_spearman": sum(1 for r in video_rows if r["num_matched_events"] >= 3 and np.isfinite(safe_float(r.get("spearman_motion_rms")))),
        "mean_pairwise_force_loudness_acc": safe_mean([safe_float(r.get("pairwise_force_loudness_acc")) for r in video_rows]),
        "mean_decay_r2": safe_mean([safe_float(r.get("decay_r2")) for r in event_rows]),
        "median_decay_r2": float(np.nanmedian([safe_float(r.get("decay_r2")) for r in event_rows])) if event_rows else float("nan"),
        "mean_decay_shape_score": safe_mean([safe_float(r.get("decay_shape_score")) for r in event_rows]),
        "median_decay_shape_score": float(np.nanmedian([safe_float(r.get("decay_shape_score")) for r in event_rows])) if event_rows else float("nan"),
        "mean_decay_lambda": safe_mean([safe_float(r.get("decay_lambda")) for r in event_rows]),
        "mean_max_drop_db": safe_mean([safe_float(r.get("max_drop_db")) for r in event_rows]),
        "mean_peak_to_floor_db": safe_mean([safe_float(r.get("peak_to_floor_db")) for r in event_rows]),
        "mean_tail_residual_mae_db": safe_mean([safe_float(r.get("tail_residual_mae_db")) for r in event_rows]),
        "mean_fit_support_ratio": safe_mean([safe_float(r.get("fit_support_ratio")) for r in event_rows]),
        "mean_spectral_flatness": safe_mean([safe_float(r.get("spectral_flatness")) for r in event_rows]),
        "mean_spectral_centroid": safe_mean([safe_float(r.get("spectral_centroid")) for r in event_rows]),
        "mean_hf_ratio": safe_mean([safe_float(r.get("hf_ratio")) for r in event_rows]),
        "mean_envelope_residual_mae": safe_mean([safe_float(r.get("envelope_residual_mae")) for r in event_rows]),
    }
    with open(metrics_dir / "source_mechanics_dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
