from __future__ import annotations

"""Build two time-contiguous action clusters and their silent Qwen clips."""

import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


def finite(value) -> float:
    try:
        result = float(value)
    except Exception:
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def first_finite(row: pd.Series, names: tuple[str, ...]) -> float:
    for name in names:
        value = finite(row.get(name))
        if math.isfinite(value):
            return value
    return float("nan")


def event_table(rows: pd.DataFrame) -> pd.DataFrame:
    events = rows.copy()
    events["event_center_sec"] = events.apply(
        lambda row: first_finite(row, ("visual_peak_sec", "audio_peak_sec")), axis=1
    )
    events["event_start_sec"] = events.apply(
        lambda row: first_finite(
            row, ("visual_start_sec", "audio_start_sec", "visual_peak_sec", "audio_peak_sec")
        ),
        axis=1,
    )
    events["event_end_sec"] = events.apply(
        lambda row: first_finite(
            row, ("visual_end_sec", "audio_end_sec", "visual_peak_sec", "audio_peak_sec")
        ),
        axis=1,
    )
    events["event_rms_db"] = pd.to_numeric(events.get("event_rms_db"), errors="coerce")
    events = events.dropna(subset=["event_center_sec", "event_rms_db"])
    return events.sort_values("event_center_sec").reset_index(drop=True)


def choose_split(events: pd.DataFrame, gap_weight: float) -> int:
    best_index = 1
    best_cost = float("inf")
    for index in range(1, len(events)):
        left = events.iloc[:index]
        right = events.iloc[index:]
        left_span = max(0.0, float(left["event_end_sec"].max() - left["event_start_sec"].min()))
        right_span = max(0.0, float(right["event_end_sec"].max() - right["event_start_sec"].min()))
        gap = float(right["event_start_sec"].min() - left["event_end_sec"].max())
        cost = left_span + right_span - gap_weight * gap
        if cost < best_cost:
            best_cost = cost
            best_index = index
    return best_index


def cluster_stats(events: pd.DataFrame, prefix: str) -> dict:
    levels = pd.to_numeric(events["event_rms_db"], errors="coerce").dropna().to_numpy(dtype=float)
    return {
        f"{prefix}_event_count": int(len(events)),
        f"{prefix}_start_sec": float(events["event_start_sec"].min()),
        f"{prefix}_end_sec": float(events["event_end_sec"].max()),
        f"{prefix}_center_sec": float(events["event_center_sec"].mean()),
        f"{prefix}_max_rms_db": float(np.max(levels)),
        f"{prefix}_mean_rms_db": float(np.mean(levels)),
        f"{prefix}_energy_sum_db": float(10.0 * np.log10(np.sum(np.power(10.0, levels / 10.0)))),
        f"{prefix}_top2_mean_rms_db": float(np.mean(np.sort(levels)[-2:])),
    }


def duration_sec(video_path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(video_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    value = finite(result.stdout.strip())
    if result.returncode or not math.isfinite(value) or value <= 0:
        raise RuntimeError(f"cannot read video duration: {video_path}")
    return value


def cut_silent_clip(video_path: Path, clip_path: Path, start: float, end: float) -> None:
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error", "-ss", f"{start:.6f}", "-i", str(video_path),
            "-t", f"{max(0.1, end - start):.6f}", "-an", "-c:v", "mpeg4", "-q:v", "2",
            str(clip_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode or not clip_path.is_file():
        raise RuntimeError(result.stderr.strip() or f"failed to create {clip_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matched-csv", type=Path, required=True)
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gap-weight", type=float, default=1.0)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    matched = pd.read_csv(args.matched_csv)
    manifest = pd.read_csv(args.manifest_csv)
    grouped = {
        str(video_id): event_table(rows)
        for video_id, rows in matched.groupby("video_id", dropna=False)
    }
    clips_root = args.output_dir / "clips"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_rows: list[dict] = []
    failures: list[dict] = []

    for _, manifest_row in manifest.iterrows():
        sample_id = str(manifest_row["video_id"])
        video_path = Path(str(manifest_row["video_path"])).resolve()
        events = grouped.get(sample_id, pd.DataFrame()).copy()
        if len(events) < 2:
            failures.append(
                {
                    "generated_sample_id": sample_id,
                    "reason": "fewer_than_two_matched_av_events",
                }
            )
            continue
        try:
            split_index = choose_split(events, args.gap_weight)
            cluster_1 = events.iloc[:split_index]
            cluster_2 = events.iloc[split_index:]
            row = {
                "generated_sample_id": sample_id,
                "source_video_path": str(video_path),
                "cluster_source": "matched_av_events",
                "event_count": int(len(events)),
                "split_index": int(split_index),
            }
            row.update(cluster_stats(cluster_1, "cluster_1"))
            row.update(cluster_stats(cluster_2, "cluster_2"))
            video_duration = duration_sec(video_path)
            boundary = float(
                np.clip(
                    0.5 * (row["cluster_1_center_sec"] + row["cluster_2_center_sec"]),
                    0.1,
                    video_duration - 0.1,
                )
            )
            cluster_dir = clips_root / sample_id
            clip_1 = cluster_dir / "cluster_1.mp4"
            clip_2 = cluster_dir / "cluster_2.mp4"
            if not args.skip_existing or not clip_1.is_file():
                cut_silent_clip(video_path, clip_1, 0.0, boundary)
            if not args.skip_existing or not clip_2.is_file():
                cut_silent_clip(video_path, clip_2, boundary, video_duration)
            row.update(
                {
                    "cluster_1_clip_path": str(clip_1.resolve()),
                    "cluster_2_clip_path": str(clip_2.resolve()),
                    "cluster_1_clip_start_sec": 0.0,
                    "cluster_1_clip_end_sec": boundary,
                    "cluster_2_clip_start_sec": boundary,
                    "cluster_2_clip_end_sec": video_duration,
                    "cluster_clip_boundary_sec": boundary,
                    "source_video_duration_sec": video_duration,
                }
            )
            output_rows.append(row)
        except Exception as exc:
            failures.append({"generated_sample_id": sample_id, "reason": str(exc)})

    pairs_jsonl = args.output_dir / "qwen_cluster_pairs.jsonl"
    with pairs_jsonl.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    pd.DataFrame(output_rows).to_csv(args.output_dir / "qwen_cluster_pairs.csv", index=False)
    pd.DataFrame(failures).to_csv(args.output_dir / "qwen_cluster_pair_failures.csv", index=False)
    summary = {
        "input_samples": int(len(manifest)),
        "prepared_pairs": int(len(output_rows)),
        "failed_pairs": int(len(failures)),
        "pairs_jsonl": str(pairs_jsonl),
    }
    (args.output_dir / "qwen_cluster_pair_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
