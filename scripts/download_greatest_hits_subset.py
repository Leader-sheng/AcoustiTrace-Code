"""Download only the Greatest Hits videos used by the 143 I2AV-LAT cases.

The official low-resolution release is a large ZIP archive.  ``remotezip``
uses HTTP range requests so this script transfers the 143 selected members
instead of downloading the complete archive.  The hosting server and any
configured proxy must support byte-range requests.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path, PurePosixPath

from remotezip import RemoteZip
from tqdm import tqdm


DEFAULT_URL = "https://web.eecs.umich.edu/~ahowens/vis/vis-data-256.zip"


def read_source_map(path: Path) -> list[dict[str, object]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != 143:
        raise ValueError(f"expected 143 source rows, found {len(rows)}")
    return rows


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-map",
        type=Path,
        default=root / "data" / "references" / "i2av_lat_sources.jsonl",
    )
    parser.add_argument("--archive-url", default=DEFAULT_URL)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "data" / "references" / "greatest_hits_videos",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    rows = read_source_map(args.source_map)
    wanted = {str(row["source_sample_id"]) for row in rows}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with RemoteZip(args.archive_url, timeout=args.timeout) as archive:
        by_stem: dict[str, list[object]] = defaultdict(list)
        for info in archive.infolist():
            member = PurePosixPath(info.filename)
            if not info.is_dir() and member.suffix.lower() in {".mp4", ".mov"}:
                by_stem[member.stem].append(info)

        missing = sorted(wanted - set(by_stem))
        ambiguous = sorted(sample_id for sample_id in wanted if len(by_stem[sample_id]) != 1)
        if missing:
            raise RuntimeError(f"official archive is missing {len(missing)} samples: {missing[:5]}")
        if ambiguous:
            raise RuntimeError(f"archive has ambiguous members for: {ambiguous[:5]}")

        for sample_id in tqdm(sorted(wanted), desc="Greatest Hits subset"):
            info = by_stem[sample_id][0]
            target_dir = args.output_dir / sample_id
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / f"{sample_id}{PurePosixPath(info.filename).suffix.lower()}"
            if target.exists() and target.stat().st_size == info.file_size and not args.overwrite:
                continue
            temporary = target.with_suffix(target.suffix + ".part")
            with archive.open(info) as source, temporary.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
            temporary.replace(target)

    downloaded = list(args.output_dir.glob("*/*"))
    print(f"ready: {len(downloaded)} Greatest Hits source videos in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
