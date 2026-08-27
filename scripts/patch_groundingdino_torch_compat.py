"""Patch the pinned GroundingDINO CUDA source for current PyTorch releases."""

from __future__ import annotations

import argparse
from pathlib import Path


RELATIVE_SOURCE = Path(
    "GroundingDINO/groundingdino/models/GroundingDINO/"
    "csrc/MsDeformAttn/ms_deform_attn_cuda.cu"
)
OLD = "AT_DISPATCH_FLOATING_TYPES(value.type(),"
NEW = "AT_DISPATCH_FLOATING_TYPES(value.scalar_type(),"
EXPECTED_OCCURRENCES = 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path("third_party/Grounded-Segment-Anything"),
        help="Grounded-Segment-Anything checkout",
    )
    args = parser.parse_args()
    source = args.repo.resolve() / RELATIVE_SOURCE
    if not source.is_file():
        raise SystemExit(f"GroundingDINO CUDA source not found: {source}")

    text = source.read_text(encoding="utf-8")
    old_count = text.count(OLD)
    new_count = text.count(NEW)
    if old_count == EXPECTED_OCCURRENCES:
        source.write_text(text.replace(OLD, NEW), encoding="utf-8")
        print(f"Patched {EXPECTED_OCCURRENCES} PyTorch dispatch calls in {source}")
        return 0
    if old_count == 0 and new_count == EXPECTED_OCCURRENCES:
        print(f"GroundingDINO compatibility patch already applied: {source}")
        return 0
    raise SystemExit(
        "Unexpected GroundingDINO source revision: "
        f"found old={old_count}, patched={new_count} dispatch calls in {source}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
