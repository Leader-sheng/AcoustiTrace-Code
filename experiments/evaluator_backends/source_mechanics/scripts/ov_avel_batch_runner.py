"""Localize visible/audio-aligned events with the public ImageBind backbone.

The paper environment used a locally modified OV-AVEL single-video entry
point. This public runner expresses the same open-vocabulary ImageBind idea
against an unmodified OV-AVEL checkout, so users do not depend on an
unpublished patch to its dataset CLI.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import math
from pathlib import Path
import subprocess
import sys

from common import ensure_dir, load_config, read_csv_dicts, write_csv_dicts


def patch_encoded_video_from_path() -> None:
    """Accept ImageBind's legacy ``sample_rate`` argument on newer pytorchvideo.

    The public ImageBind checkout forwards ``sample_rate`` to
    ``EncodedVideo.from_path``.  Recent pytorchvideo releases removed that
    keyword, even though it is irrelevant when ``decode_audio=False``.  Keep
    the compatibility handling in our runner so the external checkout can
    remain unmodified.
    """

    from pytorchvideo.data.encoded_video import EncodedVideo

    if "sample_rate" in inspect.signature(EncodedVideo.from_path).parameters:
        return
    original_from_path = EncodedVideo.from_path

    def compatible_from_path(
        file_path: str,
        decode_audio: bool = True,
        decoder: str = "pyav",
        **kwargs,
    ):
        kwargs.pop("sample_rate", None)
        return original_from_path(
            file_path,
            decode_audio=decode_audio,
            decoder=decoder,
            **kwargs,
        )

    EncodedVideo.from_path = staticmethod(compatible_from_path)


def make_windows(duration: float, window_sec: float, hop_sec: float) -> list[tuple[float, float]]:
    if not math.isfinite(duration) or duration <= 0:
        return []
    windows = []
    start = 0.0
    while start < duration:
        end = min(duration, start + window_sec)
        if end - start >= min(0.25, window_sec):
            windows.append((start, end))
        if end >= duration:
            break
        start += hop_sec
    return windows


def probe_duration(video: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(video),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr[-1000:])
    return float(result.stdout.strip())


def extract_windows(
    video: Path,
    audio: Path,
    target: Path,
    windows: list[tuple[float, float]],
) -> tuple[list[str], list[str]]:
    videos: list[str] = []
    audios: list[str] = []
    for index, (start, end) in enumerate(windows):
        visual_path = target / f"window_{index:03d}.mp4"
        audio_path = target / f"window_{index:03d}.wav"
        duration = end - start
        if not visual_path.is_file():
            command = [
                "ffmpeg", "-y", "-v", "error", "-ss", str(start), "-t", str(duration),
                "-i", str(video), "-an", "-vf", "fps=4", "-c:v", "libx264",
                "-pix_fmt", "yuv420p", str(visual_path),
            ]
            result = subprocess.run(command, text=True, capture_output=True, check=False)
            if result.returncode:
                raise RuntimeError(f"visual window extraction failed: {result.stderr[-1000:]}")
        if not audio_path.is_file():
            command = [
                "ffmpeg", "-y", "-v", "error", "-ss", str(start), "-t", str(duration),
                "-i", str(audio), "-ac", "1", "-ar", "16000", str(audio_path),
            ]
            result = subprocess.run(command, text=True, capture_output=True, check=False)
            if result.returncode:
                raise RuntimeError(f"audio window extraction failed: {result.stderr[-1000:]}")
        videos.append(str(visual_path))
        audios.append(str(audio_path))
    return videos, audios


def collapse_embeddings(value):
    import torch.nn.functional as functional

    while value.ndim > 2:
        value = value.mean(dim=1)
    return functional.normalize(value.float(), dim=-1)


def merge_detections(rows: list[dict], max_gap: float) -> list[dict]:
    merged: list[dict] = []
    for row in sorted(rows, key=lambda item: float(item["start_sec"])):
        if (
            merged
            and row["visual_label"] == merged[-1]["visual_label"]
            and float(row["start_sec"]) <= float(merged[-1]["end_sec"]) + max_gap
        ):
            previous = merged[-1]
            if float(row["confidence"]) > float(previous["confidence"]):
                previous["peak_sec"] = row["peak_sec"]
                previous["confidence"] = row["confidence"]
                previous["raw_score"] = row["raw_score"]
            previous["end_sec"] = max(float(previous["end_sec"]), float(row["end_sec"]))
        else:
            merged.append(dict(row))
    for index, row in enumerate(merged):
        row["visual_event_id"] = f"v{index:04d}"
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    settings = cfg["ov_avel"]
    ov_root = Path(settings["repo_root"])
    imagebind_root = ov_root / "proposed_method/ImageBind-main"
    if str(imagebind_root) not in sys.path:
        sys.path.insert(0, str(imagebind_root))
    import torch
    from imagebind import data as imagebind_data
    from imagebind.models import imagebind_model
    from imagebind.models.imagebind_model import ModalityType

    # ImageBind defines the tokenizer vocabulary as ``bpe/...``. Interpreting
    # that value against the process working directory only works when the
    # command happens to run from the ImageBind checkout. Evaluator adapters
    # run from the AcoustiTrace root, so anchor it to the checkout explicitly.
    bpe_path = Path(imagebind_data.BPE_PATH)
    if not bpe_path.is_absolute():
        imagebind_data.BPE_PATH = str(imagebind_root / bpe_path)

    patch_encoded_video_from_path()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("OV-AVEL/ImageBind localization requires CUDA")
    checkpoint = Path(settings["checkpoint_path"])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing ImageBind checkpoint: {checkpoint}")
    model = imagebind_model.imagebind_huge(pretrained=False)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model = model.to(device).eval()
    classes = [
        line.strip()
        for line in Path(settings["classes_file"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    with torch.inference_mode():
        text_input = imagebind_data.load_and_transform_text(
            [f"The sound of {label}." for label in classes], device
        ).to(device)
        text_embeddings = collapse_embeddings(model({ModalityType.TEXT: text_input})[ModalityType.TEXT])

    out_dir = ensure_dir(Path(settings["output_dir"]))
    raw_root = ensure_dir(out_dir / "raw")
    index_rows = read_csv_dicts(Path(cfg["output"]["root"]) / "index/videos.csv")
    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError("invalid OV-AVEL shard selection")
    index_rows = [
        row for index, row in enumerate(index_rows) if index % args.num_shards == args.shard_id
    ]
    if args.limit > 0:
        index_rows = index_rows[: args.limit]

    window_sec = float(settings.get("window_sec", 1.0))
    hop_sec = float(settings.get("hop_sec", 0.5))
    joint_threshold = float(settings.get("joint_similarity_threshold", 0.02))
    av_threshold = float(settings.get("av_similarity_threshold", 0.0))
    summary = []
    for row in index_rows:
        video_id = row["video_id"]
        video = Path(row["video_path"])
        audio = Path(row.get("audio_path") or "")
        sample_dir = ensure_dir(raw_root / video_id)
        result_csv = sample_dir / "events.csv"
        if args.skip_existing and result_csv.is_file():
            summary.append({"video_id": video_id, "status": "skipped"})
            continue
        try:
            windows = make_windows(probe_duration(video), window_sec, hop_sec)
            visual_files, audio_files = extract_windows(
                video, audio, ensure_dir(sample_dir / "windows"), windows
            )
            with torch.inference_mode():
                visual_input = imagebind_data.load_and_transform_video_data(
                    visual_files, device, clip_duration=max(1, int(math.ceil(window_sec))),
                    clips_per_video=1,
                ).to(device)
                audio_input = imagebind_data.load_and_transform_audio_data(
                    audio_files, device, clip_duration=window_sec, clips_per_video=1,
                ).to(device)
                embeddings = model(
                    {ModalityType.VISION: visual_input, ModalityType.AUDIO: audio_input}
                )
                visual_embeddings = collapse_embeddings(embeddings[ModalityType.VISION])
                audio_embeddings = collapse_embeddings(embeddings[ModalityType.AUDIO])
                visual_scores = visual_embeddings @ text_embeddings.T
                audio_scores = audio_embeddings @ text_embeddings.T
                joint_scores = 0.5 * (visual_scores + audio_scores)
                av_scores = (visual_embeddings * audio_embeddings).sum(dim=-1)

            detections = []
            for index, (start, end) in enumerate(windows):
                score, class_index = joint_scores[index].max(dim=0)
                confidence = float(score.item())
                av_confidence = float(av_scores[index].item())
                if confidence < joint_threshold or av_confidence < av_threshold:
                    continue
                detections.append(
                    {
                        "video_id": video_id,
                        "video_path": str(video),
                        "visual_event_id": "",
                        "visual_label": classes[int(class_index.item())],
                        "start_sec": start,
                        "end_sec": end,
                        "peak_sec": 0.5 * (start + end),
                        "confidence": confidence,
                        "raw_score": av_confidence,
                    }
                )
            detections = merge_detections(detections, max_gap=hop_sec + 1e-6)
            write_csv_dicts(
                result_csv,
                detections,
                [
                    "video_id", "video_path", "visual_event_id", "visual_label",
                    "start_sec", "end_sec", "peak_sec", "confidence", "raw_score",
                ],
            )
            summary.append({"video_id": video_id, "status": "ok", "num_events": len(detections)})
            del visual_input, audio_input, embeddings
        except Exception as exc:
            summary.append({"video_id": video_id, "status": "failed", "error": str(exc)[:2000]})
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    (raw_root / f"ov_avel_raw_summary_shard{args.shard_id}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
