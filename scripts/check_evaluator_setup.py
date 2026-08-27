"""Preflight the public AcoustiTrace evaluator directory and environment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def check_imports(
    python: Path,
    modules: dict[str, str],
    errors: list[str],
    *,
    pythonpath: Path | None = None,
) -> None:
    """Import a module group in an isolated interpreter.

    ``find_spec`` is insufficient for compiled packages: mismatched Torch,
    TorchAudio, CUDA, or vLLM wheels can all be discoverable but fail during
    import.  A subprocess also keeps an ABI crash from taking down preflight.
    """

    if not python.is_file():
        errors.append(f"Python interpreter not found: {python}")
        return
    probe = (
        "import importlib,json\n"
        f"mods={json.dumps(list(modules))}\n"
        "out={}\n"
        "for name in mods:\n"
        "  try: importlib.import_module(name); out[name]=''\n"
        "  except BaseException as exc: out[name]=f'{type(exc).__name__}: {exc}'\n"
        "print('ACOUSTITRACE_IMPORTS='+json.dumps(out))\n"
    )
    try:
        env = os.environ.copy()
        if pythonpath is not None:
            old_pythonpath = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = str(pythonpath) + (os.pathsep + old_pythonpath if old_pythonpath else "")
        completed = subprocess.run(
            [str(python), "-c", probe],
            text=True,
            capture_output=True,
            env=env,
            timeout=180,
            check=False,
        )
    except Exception as exc:
        errors.append(f"failed to run dependency probe with {python}: {exc}")
        return
    marker = "ACOUSTITRACE_IMPORTS="
    result_line = next(
        (line for line in reversed(completed.stdout.splitlines()) if line.startswith(marker)),
        "",
    )
    if completed.returncode != 0 or not result_line:
        detail = (completed.stderr or completed.stdout or "no diagnostic output").strip()
        errors.append(f"dependency probe failed with {python}: {detail[-1000:]}")
        return
    results = json.loads(result_line[len(marker) :])
    for module, label in modules.items():
        if results.get(module):
            errors.append(f"Python package cannot be imported: {label}: {results[module]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--skip-prompts", action="store_true")
    parser.add_argument(
        "--source-python",
        type=Path,
        default=Path(sys.executable),
        help="Python used by source-mechanics, causality, and Motion--Loudness backends",
    )
    args = parser.parse_args()
    root = args.repo_root.resolve()

    errors: list[str] = []
    warnings: list[str] = []

    for executable in ("ffmpeg", "ffprobe", "git"):
        if shutil.which(executable) is None:
            errors.append(f"executable not found on PATH: {executable}")

    main_modules = {
        "torch": "PyTorch",
        "addict": "addict (required by Grounded-SAM)",
        "cv2": "OpenCV",
        "decord": "Decord",
        "imageio": "imageio",
        "imageio_ffmpeg": "imageio-ffmpeg",
        "librosa": "librosa",
        "matplotlib": "Matplotlib",
        "numpy": "NumPy",
        "omegaconf": "OmegaConf",
        "pandas": "pandas",
        "PIL": "Pillow",
        "pycocotools": "pycocotools (required by Grounded-SAM)",
        "scipy": "SciPy",
        "soundfile": "SoundFile",
        "supervision": "Supervision (required by GroundingDINO)",
        "timm": "timm",
        "transformers": "Transformers",
        "peft": "PEFT",
        "yaml": "PyYAML",
        "yapf": "YAPF (required by GroundingDINO configs)",
    }
    source_modules = {
        "torch": "PyTorch (source evaluator environment)",
        "torchaudio": "TorchAudio (required by OV-AVEL)",
        "pytorchvideo": "PyTorchVideo (required by OV-AVEL)",
        "qwen_vl_utils": "qwen-vl-utils",
        "vllm": "vLLM (required for Motion--Loudness)",
    }
    # Keep the virtual-environment entry point intact.  Resolving the symlink
    # behind ``.venv/bin/python`` produces the system interpreter on many Linux
    # installations and makes the dependency probe inspect the wrong site-packages.
    main_python = Path(sys.executable).absolute()
    source_python = args.source_python.expanduser().absolute()
    check_imports(main_python, main_modules, errors)
    check_imports(source_python, source_modules, errors)

    groundingdino_root = (
        root / "third_party/Grounded-Segment-Anything/GroundingDINO"
    )
    if groundingdino_root.is_dir():
        check_imports(
            main_python,
            {"groundingdino._C": "compiled GroundingDINO C++/CUDA extension"},
            errors,
            pythonpath=groundingdino_root,
        )

    paths = {
        "Video-Depth-Anything repository": root / "third_party/Video-Depth-Anything",
        "Grounded-Segment-Anything repository": root / "third_party/Grounded-Segment-Anything",
        "OV-AVEL repository": root / "third_party/OV-AVEL",
        "FlexSED repository": root / "third_party/FlexSED",
        "Video-Depth-Anything checkpoint": root / "checkpoints/video_depth_anything/metric_video_depth_anything_vitl.pth",
        "GroundingDINO checkpoint": root / "checkpoints/grounded_sam/groundingdino_swint_ogc.pth",
        "SAM checkpoint": root / "checkpoints/grounded_sam/sam_vit_b_01ec64.pth",
        "ImageBind checkpoint": root / "checkpoints/ov_avel/imagebind_huge.pth",
        "FlexSED checkpoint": root / "checkpoints/flexsed/flexsed_as.pt",
        "FlexSED CLAP config": root / "checkpoints/flexsed/laion-clap-htsat-unfused/config.json",
        "FlexSED CLAP weights": root / "checkpoints/flexsed/laion-clap-htsat-unfused/pytorch_model.bin",
        "Qwen3-VL base model": root / "checkpoints/qwen3-vl/config.json",
        "RT60 physics head": root / "checkpoints/acoustitrace-rt60/physics_head.pt",
        "RT60 adapter config": root / "checkpoints/acoustitrace-rt60/vlm_adapter/adapter_config.json",
    }
    if not args.skip_prompts:
        paths.update(
            {
                "T2AV release manifest": root / "data/prompts/t2av_605.jsonl",
                "I2AV-LAT release manifest": root / "data/prompts/i2av_lat_143.jsonl",
            }
        )
    for label, path in paths.items():
        if not path.exists():
            errors.append(f"missing {label}: {path}")

    adapter_dir = root / "checkpoints/acoustitrace-rt60/vlm_adapter"
    if adapter_dir.is_dir() and not any(
        (adapter_dir / name).is_file()
        for name in ("adapter_model.safetensors", "adapter_model.bin")
    ):
        errors.append(f"missing RT60 LoRA weights in {adapter_dir}")

    try:
        import torch

        if not torch.cuda.is_available():
            warnings.append("torch.cuda.is_available() is false; native GPU backends will not run")
        else:
            print(f"OK  CUDA: {torch.cuda.get_device_name(0)}")
    except Exception as exc:
        warnings.append(f"main-environment CUDA check failed: {exc}")

    if errors:
        for item in errors:
            print(f"ERR {item}")
    for item in warnings:
        print(f"WARN {item}")
    if errors:
        print(f"\nPreflight failed with {len(errors)} error(s).")
        return 1
    print("OK  AcoustiTrace evaluator setup is complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
