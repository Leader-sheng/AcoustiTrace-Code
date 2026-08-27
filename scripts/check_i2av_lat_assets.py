"""Validate the bundled I2AV Log Attack Time conditioning and audio assets."""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
EXPECTED_COUNT = 143
EXPECTED_CHANNELS = 1
EXPECTED_SAMPLE_RATE = 22_050
EXPECTED_SAMPLE_WIDTH = 2
EXPECTED_DURATION_SEC = 5.0


def validate_assets(root: Path, manifest: Path) -> list[str]:
    root = root.resolve()
    manifest = manifest.resolve()
    rows = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    errors: list[str] = []
    if len(rows) != EXPECTED_COUNT:
        errors.append(f"expected {EXPECTED_COUNT} manifest rows, found {len(rows)}")

    prompt_ids: set[str] = set()
    conditioning_paths: set[Path] = set()
    reference_paths: set[Path] = set()
    for row in rows:
        prompt_id = str(row.get("prompt_id", ""))
        if not prompt_id:
            errors.append("manifest row has no prompt_id")
            continue
        if prompt_id in prompt_ids:
            errors.append(f"duplicate prompt_id: {prompt_id}")
        prompt_ids.add(prompt_id)

        conditioning = (manifest.parent / str(row["conditioning_asset_path"])).resolve()
        reference = (
            root
            / str(row["evaluator_inputs"]["log_attack_time"]["reference_audio_path"])
        ).resolve()
        if not conditioning.is_relative_to(root):
            errors.append(f"{prompt_id}: conditioning path escapes repository root")
            continue
        if not reference.is_relative_to(root):
            errors.append(f"{prompt_id}: reference path escapes repository root")
            continue
        conditioning_paths.add(conditioning)
        reference_paths.add(reference)

        if not conditioning.is_file():
            errors.append(f"{prompt_id}: missing conditioning PNG")
        elif conditioning.read_bytes()[: len(PNG_SIGNATURE)] != PNG_SIGNATURE:
            errors.append(f"{prompt_id}: invalid conditioning PNG signature")

        if not reference.is_file():
            errors.append(f"{prompt_id}: missing reference WAV")
            continue
        try:
            with wave.open(str(reference), "rb") as handle:
                channels = handle.getnchannels()
                sample_width = handle.getsampwidth()
                sample_rate = handle.getframerate()
                frames = handle.getnframes()
        except (EOFError, wave.Error) as exc:
            errors.append(f"{prompt_id}: unreadable reference WAV: {exc}")
            continue
        if channels != EXPECTED_CHANNELS:
            errors.append(f"{prompt_id}: expected mono WAV, found {channels} channels")
        if sample_width != EXPECTED_SAMPLE_WIDTH:
            errors.append(
                f"{prompt_id}: expected 16-bit PCM WAV, found {sample_width * 8}-bit samples"
            )
        if sample_rate != EXPECTED_SAMPLE_RATE:
            errors.append(
                f"{prompt_id}: expected {EXPECTED_SAMPLE_RATE} Hz, found {sample_rate} Hz"
            )
        duration = frames / sample_rate if sample_rate else 0.0
        if abs(duration - EXPECTED_DURATION_SEC) > 1.0 / EXPECTED_SAMPLE_RATE:
            errors.append(f"{prompt_id}: expected 5.0 s WAV, found {duration:.6f} s")

    if len(conditioning_paths) != EXPECTED_COUNT:
        errors.append(
            f"expected {EXPECTED_COUNT} unique conditioning paths, found {len(conditioning_paths)}"
        )
    if len(reference_paths) != EXPECTED_COUNT:
        errors.append(
            f"expected {EXPECTED_COUNT} unique reference paths, found {len(reference_paths)}"
        )
    return errors


def main() -> int:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=default_root / "data" / "prompts" / "i2av_lat_143.jsonl",
    )
    args = parser.parse_args()
    errors = validate_assets(args.root, args.manifest)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        print(f"failed: {len(errors)} bundled-asset validation error(s)")
        return 1
    print("ready: 143 conditioning PNGs and 143 five-second reference WAVs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
