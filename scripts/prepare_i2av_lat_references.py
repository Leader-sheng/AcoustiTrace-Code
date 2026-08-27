"""Prepare I2AV conditioning frames and five-second Greatest Hits references."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def read_source_map(path: Path) -> list[dict[str, object]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != 143:
        raise ValueError(f"expected 143 source rows, found {len(rows)}")
    return rows


def video_index(root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    duplicates: set[str] = set()
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".mp4", ".mov"}:
            if path.stem in index:
                duplicates.add(path.stem)
            index[path.stem] = path
    if duplicates:
        raise ValueError(f"duplicate source video stems: {sorted(duplicates)[:5]}")
    return index


def perceptual_hash(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("empty image")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
    coefficients = cv2.dct(resized.astype(np.float32))[:8, :8]
    threshold = float(np.median(coefficients.reshape(-1)[1:]))
    return coefficients > threshold


def read_video_frame(video_path: Path, frame_index: int) -> np.ndarray:
    if frame_index < 0:
        raise ValueError("conditioning frame index must be non-negative")
    capture = cv2.VideoCapture(str(video_path))
    frame: np.ndarray | None = None
    ok = False
    for _ in range(frame_index + 1):
        ok, frame = capture.read()
        if not ok:
            break
    capture.release()
    if not ok or frame is None:
        raise ValueError(f"cannot read frame {frame_index} from video: {video_path}")
    return frame


def conditioning_frame_distance(
    video_path: Path,
    image_path: Path,
    frame_index: int,
) -> float:
    reference = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if reference is None:
        raise ValueError(f"cannot read conditioning image: {image_path}")
    frame = read_video_frame(video_path, frame_index)
    return float(np.mean(perceptual_hash(reference) != perceptual_hash(frame)))


def write_conditioning_frame(
    video_path: Path,
    image_path: Path,
    frame_index: int,
) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    frame = read_video_frame(video_path, frame_index)
    if not cv2.imwrite(str(image_path), frame):
        raise ValueError(f"cannot write conditioning image: {image_path}")


def probe_duration(path: Path, ffprobe: str) -> float:
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    duration = float(result.stdout.strip())
    if not math.isfinite(duration):
        raise ValueError(f"invalid duration for {path}")
    return duration


def extract_audio(source: Path, target: Path, duration: float, ffmpeg: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-i",
            str(source),
            "-t",
            f"{duration:.6f}",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "22050",
            "-c:a",
            "pcm_s16le",
            str(target),
        ],
        check=True,
    )


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-map",
        type=Path,
        default=root / "data" / "references" / "i2av_lat_sources.jsonl",
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=root / "data" / "references" / "greatest_hits_videos",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "data" / "references" / "i2av_lat",
    )
    parser.add_argument("--report", type=Path, default=root / "outputs" / "i2av_lat_reference_report.csv")
    parser.add_argument(
        "--conditioning-frame-index",
        type=int,
        default=1,
        help="zero-based decoded frame used as the I2AV condition (default: second frame)",
    )
    parser.add_argument(
        "--create-missing-conditioning-images",
        action="store_true",
        help="write missing conditioning PNGs from the selected source-video frame",
    )
    parser.add_argument(
        "--max-conditioning-frame-distance",
        "--max-first-frame-distance",
        dest="max_conditioning_frame_distance",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--skip-conditioning-frame-check",
        "--skip-first-frame-check",
        dest="skip_conditioning_frame_check",
        action="store_true",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args()

    rows = read_source_map(args.source_map)
    sources = video_index(args.videos_dir)
    report_rows: list[dict[str, object]] = []
    failures = 0

    for row in tqdm(rows, desc="I2AV-LAT references"):
        prompt_id = str(row["prompt_id"])
        sample_id = str(row["source_sample_id"])
        target = args.output_dir / f"{prompt_id}.wav"
        report: dict[str, object] = {
            "prompt_id": prompt_id,
            "source_sample_id": sample_id,
            "source_video_path": "",
            "source_duration_sec": "",
            "conditioning_frame_index": args.conditioning_frame_index,
            "conditioning_frame_phash_distance": "",
            "conditioning_frame_phash_threshold": "",
            "selected_event_start_sec": row.get("selected_event_start_sec"),
            "selected_event_end_sec": row.get("selected_event_end_sec"),
            "reference_audio_path": str(target),
            "status": "failed",
            "error": "",
        }
        try:
            source = sources[sample_id]
            report["source_video_path"] = str(source)
            duration = probe_duration(source, args.ffprobe)
            report["source_duration_sec"] = duration
            clip_duration = float(row.get("target_clip_duration_sec", 5.0))
            if duration + 0.02 < clip_duration:
                raise ValueError(f"source is only {duration:.3f}s; need {clip_duration:.3f}s")

            condition = root / str(row["conditioning_asset_path"])
            if not condition.is_file() and args.create_missing_conditioning_images:
                write_conditioning_frame(source, condition, args.conditioning_frame_index)
            if not args.skip_conditioning_frame_check:
                max_distance = float(
                    row.get(
                        "max_conditioning_frame_distance",
                        args.max_conditioning_frame_distance,
                    )
                )
                report["conditioning_frame_phash_threshold"] = max_distance
                distance = conditioning_frame_distance(
                    source,
                    condition,
                    args.conditioning_frame_index,
                )
                report["conditioning_frame_phash_distance"] = distance
                if distance > max_distance:
                    raise ValueError(
                        f"conditioning-frame mismatch: pHash distance {distance:.3f} exceeds "
                        f"{max_distance:.3f}"
                    )

            if args.overwrite or not target.exists():
                extract_audio(source, target, clip_duration, args.ffmpeg)
            report["status"] = "ready"
        except Exception as exc:
            failures += 1
            report["error"] = str(exc)
        report_rows.append(report)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report_rows[0]))
        writer.writeheader()
        writer.writerows(report_rows)

    ready = len(report_rows) - failures
    print(f"ready: {ready}/143 reference audios; report: {args.report}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
