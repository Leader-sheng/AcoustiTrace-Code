"""Check the public visual RT60 checkpoint contract without loading the model."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    root = args.checkpoint.resolve()
    errors: list[str] = []
    if not root.is_dir():
        errors.append(f"checkpoint directory does not exist: {root}")
    if not (root / "physics_head.pt").is_file():
        errors.append("missing physics_head.pt")
    adapter = root / "vlm_adapter"
    if not adapter.is_dir():
        errors.append("missing vlm_adapter/")
    elif not (adapter / "adapter_config.json").is_file():
        errors.append("missing vlm_adapter/adapter_config.json")
    adapter_weights = [adapter / "adapter_model.safetensors", adapter / "adapter_model.bin"]
    if adapter.is_dir() and not any(path.is_file() for path in adapter_weights):
        errors.append("missing LoRA adapter weights")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"OK: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
