from __future__ import annotations

import argparse
import csv
import math
import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np
from scipy.ndimage import gaussian_filter1d


SAMPLE_RATE = 22050
FRAME_LENGTH = 1024
HOP_LENGTH = 128
SMOOTH_SIGMA = 1.0
ATTACK_PRE_SEC = 0.05
ATTACK_POST_SEC = 0.30
BASELINE_SEC = 0.05
DURATION_FLOOR_SEC = 1e-4
SCORE_SCALE = 0.35


def load_audio(path: str | Path) -> np.ndarray:
    source = Path(path)
    if source.suffix.lower() in {".wav", ".flac", ".mp3", ".ogg"}:
        audio, _ = librosa.load(source, sr=SAMPLE_RATE, mono=True)
        return audio
    with tempfile.NamedTemporaryFile(suffix=".wav") as handle:
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error", "-i", str(source),
                "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), handle.name,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "audio extraction failed")
        audio, _ = librosa.load(handle.name, sr=SAMPLE_RATE, mono=True)
        return audio


def estimate_attack_duration(audio: np.ndarray, onset_sec: float | None = None) -> float:
    if audio.size < FRAME_LENGTH:
        raise ValueError("insufficient audio support")
    rms = librosa.feature.rms(
        y=audio, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH
    )[0]
    envelope = gaussian_filter1d(rms.astype(float), sigma=SMOOTH_SIGMA)
    times = librosa.frames_to_time(
        np.arange(envelope.size), sr=SAMPLE_RATE, hop_length=HOP_LENGTH
    )
    if onset_sec is None or not math.isfinite(onset_sec):
        strength = librosa.onset.onset_strength(
            y=audio, sr=SAMPLE_RATE, hop_length=HOP_LENGTH
        )
        onset_sec = float(
            librosa.frames_to_time(int(np.argmax(strength)), sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
        )

    baseline_mask = (times >= max(0.0, onset_sec - BASELINE_SEC)) & (times < onset_sec)
    attack_mask = (times >= max(0.0, onset_sec - ATTACK_PRE_SEC)) & (
        times <= onset_sec + ATTACK_POST_SEC
    )
    attack_indices = np.flatnonzero(attack_mask)
    if attack_indices.size < 3:
        raise ValueError("insufficient attack frames")
    peak_index = int(attack_indices[np.argmax(envelope[attack_indices])])
    rise_indices = attack_indices[attack_indices <= peak_index]
    baseline = (
        float(np.percentile(envelope[baseline_mask], 25))
        if baseline_mask.any()
        else float(np.min(envelope[rise_indices]))
    )
    amplitude = float(envelope[peak_index] - baseline)
    if amplitude <= 1e-8:
        raise ValueError("no stable baseline-to-peak range")
    t10 = rise_indices[envelope[rise_indices] >= baseline + 0.1 * amplitude]
    t90 = rise_indices[envelope[rise_indices] >= baseline + 0.9 * amplitude]
    if not t10.size or not t90.size:
        raise ValueError("attack threshold crossing missing")
    duration = float(times[int(t90[0])] - times[int(t10[0])])
    if duration < 0:
        raise ValueError("invalid attack threshold order")
    return max(duration, DURATION_FLOOR_SEC)


def score_pair(generated_duration: float, reference_duration: float) -> float:
    error = abs(math.log(generated_duration) - math.log(reference_duration))
    return 100.0 * math.exp(-error / SCORE_SCALE)


def optional_float(row: dict[str, str], key: str) -> float | None:
    try:
        value = float(row.get(key, ""))
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute the paper-aligned I2AV Log Attack Time score."
    )
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.pairs, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    output_rows = []
    for row in rows:
        result = {"sample_id": row.get("sample_id", ""), "valid": False, "score": ""}
        try:
            generated = estimate_attack_duration(
                load_audio(row["generated_path"]),
                optional_float(row, "generated_onset_sec"),
            )
            reference = estimate_attack_duration(
                load_audio(row["reference_path"]),
                optional_float(row, "reference_onset_sec"),
            )
            result.update(
                {
                    "valid": True,
                    "generated_attack_sec": generated,
                    "reference_attack_sec": reference,
                    "log_attack_abs_error": abs(math.log(generated) - math.log(reference)),
                    "score": score_pair(generated, reference),
                    "error": "",
                }
            )
        except Exception as exc:
            result["error"] = str(exc)
        output_rows.append(result)

    fields = [
        "sample_id", "valid", "generated_attack_sec", "reference_attack_sec",
        "log_attack_abs_error", "score", "error",
    ]
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)


if __name__ == "__main__":
    main()
