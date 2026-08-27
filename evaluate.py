"""Convenience entry point for the public AcoustiTrace evaluator."""

import sys
from acoustitrace.cli import main


if __name__ == "__main__":
    sys.argv.insert(1, "evaluate")
    raise SystemExit(main())
