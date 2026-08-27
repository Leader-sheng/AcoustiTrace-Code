"""Download a versioned AcoustiTrace checkpoint from Hugging Face Hub."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-id",
        default=os.environ.get("ACOUSTITRACE_RT60_REPO_ID"),
        help="Hugging Face repository ID (or set ACOUSTITRACE_RT60_REPO_ID)",
    )
    parser.add_argument("--revision", default="main")
    parser.add_argument("--local-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.repo_id:
        raise SystemExit("--repo-id or ACOUSTITRACE_RT60_REPO_ID is required")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "Install huggingface-hub: python -m pip install huggingface-hub"
        ) from exc
    destination = snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=args.local_dir,
    )
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
