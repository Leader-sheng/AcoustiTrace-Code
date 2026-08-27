"""Run audio/visual RT60 inference and emit the public evaluator protocol.

The adapter deliberately keeps preprocessing, the audio Schroeder proxy, and
the learned visual physics head behind one command.  It operates on generated
videos only; training annotations and ground-truth RT60 values are never read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

from common import finite, native_id, python_stage, read_jsonl, run_stage, write_jsonl


def extract_media(video: Path, sample_dir: Path, frame_time: float) -> dict[str, str]:
    """Extract audio and construct the three visual inputs used in training."""

    import cv2
    from PIL import Image

    sample_dir.mkdir(parents=True, exist_ok=True)
    audio_path = sample_dir / "audio.wav"
    rgb_path = sample_dir / "rgb.png"
    depth_path = sample_dir / "depth.png"
    alpha_path = sample_dir / "alpha.png"
    alpha_map_path = sample_dir / "alpha_map.npy"

    if not audio_path.is_file():
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error", "-i", str(video), "-vn",
                "-ac", "1", "-ar", "24000", str(audio_path),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode:
            raise RuntimeError(f"audio extraction failed: {result.stderr[-2000:]}")

    if not rgb_path.is_file():
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise RuntimeError("cannot open generated video")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = frames / fps if fps > 0 and frames > 0 else 0.0
        actual_time = min(frame_time, max(0.0, duration - 0.05)) if duration else frame_time
        if fps > 0:
            capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(round(actual_time * fps))))
        ok, frame = capture.read()
        capture.release()
        if not ok or frame is None or not cv2.imwrite(str(rgb_path), frame):
            raise RuntimeError("failed to extract RT60 visual frame")

    if not depth_path.is_file():
        rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb_bgr is None:
            raise RuntimeError("failed to read extracted RGB frame")
        gray = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY)
        pseudo_depth = 255 - cv2.GaussianBlur(gray, (9, 9), 0)
        if not cv2.imwrite(str(depth_path), pseudo_depth):
            raise RuntimeError("failed to write pseudo-depth input")

    if not alpha_path.is_file() or not alpha_map_path.is_file():
        rgb = np.asarray(Image.open(rgb_path).convert("RGB").resize((512, 512)), dtype=np.float32) / 255.0
        brightness = rgb.mean(axis=2)
        saturation = rgb.max(axis=2) - rgb.min(axis=2)
        alpha = 0.04 + 0.20 * saturation + 0.10 * (
            1.0 - np.abs(brightness - 0.55) / 0.55
        )
        alpha = np.clip(alpha, 0.03, 0.45).astype(np.float32)
        np.save(alpha_map_path, alpha)
        # Use a deterministic perceptual rendering without a matplotlib border.
        normalized = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_VIRIDIS)
        if not cv2.imwrite(str(alpha_path), colored):
            raise RuntimeError("failed to write acoustic-alpha input")

    return {
        "audio": str(audio_path),
        "rgb": str(rgb_path),
        "depth": str(depth_path),
        "alpha": str(alpha_path),
        "alpha_map": str(alpha_map_path),
    }


def patch_processor_check() -> None:
    try:
        from transformers.processing_utils import ProcessorMixin

        ProcessorMixin.check_argument_for_proper_class = lambda self, attribute_name, arg: None
    except Exception:
        pass


def compatible_adapter_path(path: Path) -> str:
    """Return an adapter directory and fail early on incomplete downloads."""

    config = path / "adapter_config.json"
    weights = [path / "adapter_model.safetensors", path / "adapter_model.bin"]
    if not config.is_file() or not any(item.is_file() for item in weights):
        raise FileNotFoundError(
            f"incomplete RT60 VLM adapter at {path}; expected adapter_config.json "
            "and adapter_model.safetensors (or .bin)"
        )
    return str(path)


class VisualPhysicsHead:
    """Minimal inference wrapper for the released Sabine-guided head."""

    def __init__(self, repo_root: Path, base_model: Path, checkpoint: Path) -> None:
        import torch
        import yaml

        backend_root = repo_root / "experiments/evaluator_backends/rt60"
        sys.path.insert(0, str(backend_root))
        from visual_physics_runtime import RT60_500_PROMPT, build_vlm_and_processor

        if not (checkpoint / "physics_head.pt").is_file():
            raise FileNotFoundError(f"missing {checkpoint / 'physics_head.pt'}")
        if not base_model.is_dir():
            raise FileNotFoundError(f"missing Qwen3-VL base model directory: {base_model}")

        config_path = backend_root / "rt60_runtime.yaml"
        config: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config["model"]["model_name_or_path"] = str(base_model)
        adapter = checkpoint / "vlm_adapter"
        config["model"]["init_lora_path"] = compatible_adapter_path(adapter)
        config["model"]["enable_lora"] = False
        config["model"]["freeze_vlm"] = True
        config["model"]["train_vlm_adapter"] = False
        # SDPA is more portable than FlashAttention for the first public setup.
        config["model"]["attn_implementation"] = "sdpa"
        config["model"]["gradient_checkpointing"] = False

        patch_processor_check()
        self.torch = torch
        self.prompt = RT60_500_PROMPT
        self.model, self.processor = build_vlm_and_processor(config)
        state = torch.load(checkpoint / "physics_head.pt", map_location="cpu", weights_only=True)
        self.model.physics_head.load_state_dict(state)
        local_rank = int(__import__("os").environ.get("LOCAL_RANK", "0"))
        self.device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def predict(self, images: list[str]) -> float:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": images[0]},
                    {"type": "image", "image": images[1]},
                    {"type": "image", "image": images[2]},
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        value = float(outputs["rt60_500Hz_pred"][0].float().item())
        if finite(value) is None or not 0.05 <= value <= 5.0:
            raise ValueError(f"visual RT60 prediction out of range: {value}")
        return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--frame-time", type=float, default=0.8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    work_dir = Path(args.work_dir).resolve()
    native_root = work_dir / "native"
    jobs = read_jsonl(args.input)
    id_map = {native_id(job["sample_id"]): job for job in jobs}
    media: dict[str, dict[str, str]] = {}
    early_failures: dict[str, str] = {}
    for identifier, job in id_map.items():
        try:
            media[identifier] = extract_media(
                Path(job["video_path"]), native_root / "inputs" / identifier, args.frame_time
            )
        except Exception as exc:
            early_failures[identifier] = str(exc)

    audio_manifest = native_root / "audio_manifest.jsonl"
    write_jsonl(
        audio_manifest,
        [
            {"sample_id": identifier, "audio_path": paths["audio"]}
            for identifier, paths in media.items()
        ],
    )
    audio_root = native_root / "audio_rt60"
    audio_json = audio_root / "outputs/audio_rt60_500_proxy.jsonl"
    if media and not (args.resume and audio_json.is_file()):
        run_stage(
            python_stage(
                repo_root / "experiments/evaluator_backends/rt60/audio_rt60_proxy.py",
                "--output_root", str(audio_root),
                "--manifest", str(audio_manifest),
                "--target_sr", "24000",
                "--center_hz", "500",
                "--input_format", "mono",
            ),
            cwd=repo_root,
        )
    audio_rows = {
        row["sample_id"]: row for row in read_jsonl(audio_json)
    } if audio_json.is_file() else {}

    predictor = VisualPhysicsHead(
        repo_root, Path(args.base_model).resolve(), Path(args.checkpoint).resolve()
    ) if media else None
    output_rows = []
    for identifier, job in id_map.items():
        if identifier in early_failures:
            output_rows.append(
                {
                    "sample_id": job["sample_id"],
                    "evaluator": "rt60_consistency",
                    "status": "invalid",
                    "reason": early_failures[identifier],
                    "evidence": {},
                    "backend_version": "rt60-native-v1",
                }
            )
            continue
        audio = audio_rows.get(identifier, {})
        audio_value = finite(audio.get("audio_apparent_rt60_500_proxy"))
        if not audio.get("audio_rt_valid") or audio_value is None:
            output_rows.append(
                {
                    "sample_id": job["sample_id"],
                    "evaluator": "rt60_consistency",
                    "status": "invalid",
                    "reason": str(audio.get("audio_rejection_reason", "invalid audio RT60 proxy")),
                    "evidence": {},
                    "backend_version": "rt60-native-v1",
                }
            )
            continue
        try:
            paths = media[identifier]
            assert predictor is not None
            visual_value = predictor.predict([paths["rgb"], paths["depth"], paths["alpha"]])
            output_rows.append(
                {
                    "sample_id": job["sample_id"],
                    "evaluator": "rt60_consistency",
                    "status": "success",
                    "reason": "",
                    "evidence": {"audio_rt60": audio_value, "visual_rt60": visual_value},
                    "backend_version": "rt60-native-v1",
                    "artifacts": {
                        "audio_diagnostic": audio.get("diagnostic_path", ""),
                        "rgb": paths["rgb"],
                        "depth": paths["depth"],
                        "alpha": paths["alpha"],
                    },
                }
            )
        except Exception as exc:
            output_rows.append(
                {
                    "sample_id": job["sample_id"],
                    "evaluator": "rt60_consistency",
                    "status": "invalid",
                    "reason": f"visual RT60 failed: {exc}",
                    "evidence": {},
                    "backend_version": "rt60-native-v1",
                }
            )
    write_jsonl(args.output, output_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
