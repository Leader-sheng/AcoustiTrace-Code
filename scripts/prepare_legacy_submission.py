"""Convert legacy AcoustiTrace result ZIPs to the canonical submission layout."""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path, PurePosixPath

from tqdm import tqdm


def read_prompt_ids(path: Path, expected_task: str) -> set[str]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    prompt_ids = {
        str(row["prompt_id"])
        for row in rows
        if str(row.get("task", "")).strip() == expected_task
    }
    if len(prompt_ids) != len(rows):
        raise ValueError(f"{path} contains duplicate IDs or rows outside task={expected_task}")
    return prompt_ids


def legacy_prompt_id(member_name: str) -> str | None:
    member = PurePosixPath(member_name)
    if member.suffix.lower() != ".mp4":
        return None
    parts = member.stem.split("__")
    if len(parts) < 3 or not parts[0].isdigit():
        return None
    prompt_id = "__".join(parts[1:-1]).strip()
    return prompt_id or None


def index_archive(archive: zipfile.ZipFile, expected: set[str]) -> dict[str, zipfile.ZipInfo]:
    matches: dict[str, zipfile.ZipInfo] = {}
    duplicates: set[str] = set()
    for info in archive.infolist():
        prompt_id = legacy_prompt_id(info.filename)
        if prompt_id not in expected:
            continue
        if prompt_id in matches:
            duplicates.add(prompt_id)
        matches[prompt_id] = info
    if duplicates:
        raise ValueError(f"archive contains duplicate prompt IDs: {sorted(duplicates)[:5]}")
    missing = sorted(expected - set(matches))
    if missing:
        raise ValueError(f"archive is missing {len(missing)} expected prompts: {missing[:5]}")
    return matches


def extract_archive(
    archive_path: Path,
    expected: set[str],
    output_dir: Path,
    overwrite: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        members = index_archive(archive, expected)
        for prompt_id in tqdm(sorted(expected), desc=output_dir.name):
            target = output_dir / f"{prompt_id}.mp4"
            info = members[prompt_id]
            if target.is_file() and target.stat().st_size == info.file_size and not overwrite:
                continue
            temporary = target.with_suffix(".part.mp4")
            with archive.open(info) as source, temporary.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
            temporary.replace(target)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t2av-zip", type=Path, required=True)
    parser.add_argument("--i2av-zip", type=Path, required=True)
    parser.add_argument(
        "--t2av-prompts",
        type=Path,
        default=root / "data" / "prompts" / "t2av_605.jsonl",
    )
    parser.add_argument(
        "--i2av-prompts",
        type=Path,
        default=root / "data" / "prompts" / "i2av_lat_143.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "submissions" / "LTX-2.3",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    t2av_ids = read_prompt_ids(args.t2av_prompts, "t2av")
    i2av_ids = read_prompt_ids(args.i2av_prompts, "i2av")
    if len(t2av_ids) != 605:
        raise ValueError(f"expected 605 T2AV prompts, found {len(t2av_ids)}")
    if len(i2av_ids) != 143:
        raise ValueError(f"expected 143 I2AV-LAT prompts, found {len(i2av_ids)}")

    extract_archive(args.t2av_zip, t2av_ids, args.output_dir / "t2av", args.overwrite)
    extract_archive(args.i2av_zip, i2av_ids, args.output_dir / "i2av_lat", args.overwrite)
    print(f"ready: 605 T2AV + 143 I2AV-LAT videos in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
