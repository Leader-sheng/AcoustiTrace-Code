"""Run OV-AVEL/FlexSED/Qwen and adapt Motion--Loudness/Impact evidence."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

import yaml

from common import finite, native_id, python_stage, read_csv, read_jsonl, run_stage, write_csv, write_jsonl


def extract_audio(video: Path, audio: Path) -> None:
    audio.parent.mkdir(parents=True, exist_ok=True)
    if audio.is_file():
        return
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error", "-i", str(video), "-vn",
            "-ac", "1", "-ar", "22050", str(audio),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"audio extraction failed for {video}: {result.stderr}")


def boolean(value) -> bool | None:
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--qwen-checkpoint", default="")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    work_dir = Path(args.work_dir).resolve()
    native_root = work_dir / "native"
    jobs = read_jsonl(args.input)
    id_map = {native_id(job["sample_id"]): job for job in jobs}
    index_rows = []
    audio_root = native_root / "audio"
    for identifier, job in id_map.items():
        audio = audio_root / f"{identifier}.wav"
        extract_audio(Path(job["video_path"]), audio)
        index_rows.append(
            {"video_id": identifier, "video_path": job["video_path"], "audio_path": str(audio)}
        )
    index_path = native_root / "index/videos.csv"
    write_csv(index_path, index_rows, ("video_id", "video_path", "audio_path"))
    motion_index_rows = [
        row for row in index_rows
        if "motion_loudness" in id_map[row["video_id"]]["evaluators"]
    ]
    motion_index_path = native_root / "index/motion_videos.csv"
    write_csv(motion_index_path, motion_index_rows, ("video_id", "video_path", "audio_path"))

    backend = repo_root / "experiments/evaluator_backends/source_mechanics"
    scripts = backend / "scripts"
    base_config = yaml.safe_load(
        (backend / "configs/source_mechanics_config.yaml").read_text(encoding="utf-8")
    )
    base_config["data"]["review_package_root"] = str(native_root)
    base_config["output"]["root"] = str(native_root)
    base_config["ov_avel"]["repo_root"] = str(repo_root / "third_party/OV-AVEL")
    base_config["ov_avel"]["classes_file"] = str(backend / "configs/ov_avel_classes.txt")
    base_config["ov_avel"]["output_dir"] = str(native_root / "ov_avel")
    base_config["ov_avel"]["checkpoint_path"] = str(
        repo_root / "checkpoints/ov_avel/imagebind_huge.pth"
    )
    base_config["flexsed"]["repo_root"] = str(repo_root / "third_party/FlexSED")
    base_config["flexsed"]["output_dir"] = str(native_root / "flexsed")
    base_config["flexsed"]["checkpoint_path"] = str(
        repo_root / "checkpoints/flexsed/flexsed_as.pt"
    )
    base_config["flexsed"]["clap_model_path"] = str(
        repo_root / "checkpoints/flexsed/laion-clap-htsat-unfused"
    )
    config_path = native_root / "source_mechanics_config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(base_config, sort_keys=False), encoding="utf-8")

    ov_dir = native_root / "ov_avel"
    ov_dir.mkdir(parents=True, exist_ok=True)
    with (ov_dir / "ov_avel_video_list.txt").open("w", encoding="utf-8") as handle:
        for row in index_rows:
            handle.write(f"{row['video_id']}\t{row['video_path']}\t{row['audio_path']}\n")
    resume_flag = ["--skip-existing"] if args.resume else []
    stages = [
        python_stage(scripts / "ov_avel_batch_runner.py", "--config", str(config_path), *resume_flag),
        python_stage(scripts / "flexsed_batch_runner.py", "--config", str(config_path), *resume_flag),
        python_stage(scripts / "04_parse_ov_avel_outputs.py", "--config", str(config_path)),
        python_stage(scripts / "05_parse_flexsed_outputs.py", "--config", str(config_path)),
        python_stage(scripts / "06_match_av_events.py", "--config", str(config_path)),
        python_stage(scripts / "07_compute_source_mechanics_metrics.py", "--config", str(config_path)),
    ]
    for stage in stages:
        run_stage(stage, cwd=repo_root)

    motion_config = yaml.safe_load(
        (backend / "configs/motion_loudness_config.yaml").read_text(encoding="utf-8")
    )
    motion_root = native_root / "motion_loudness"
    qwen_checkpoint = Path(args.qwen_checkpoint).resolve() if args.qwen_checkpoint else repo_root / "checkpoints/qwen3-vl"
    motion_config["qwen3"]["checkpoint_path"] = str(qwen_checkpoint)
    motion_config_path = native_root / "motion_loudness_config.yaml"
    motion_config_path.write_text(yaml.safe_dump(motion_config, sort_keys=False), encoding="utf-8")
    pair_root = motion_root / "cluster_pairs"
    qwen_root = motion_root / "qwen_motion_only"
    final_root = motion_root / "frame_diff_fallback"
    needs_motion = any("motion_loudness" in job["evaluators"] for job in jobs)
    motion_error = ""
    if needs_motion:
        try:
            run_stage(
                python_stage(
                    scripts / "09_prepare_motion_loudness_cluster_pairs.py",
                    "--matched-csv", str(native_root / "metrics/source_mechanics_event_metrics.csv"),
                    "--manifest-csv", str(motion_index_path),
                    "--output-dir", str(pair_root),
                    "--gap-weight", str(motion_config["clustering"]["gap_weight"]),
                    *(["--skip-existing"] if args.resume else []),
                ),
                cwd=repo_root,
            )
            run_stage(
                python_stage(
                    scripts / "10_qwen3_motion_loudness_cluster_score.py",
                    "--pairs-jsonl", str(pair_root / "qwen_cluster_pairs.jsonl"),
                    "--checkpoint-path", str(qwen_checkpoint),
                    "--output-dir", str(qwen_root),
                    "--gpu-memory-utilization", str(motion_config["qwen3"]["gpu_memory_utilization"]),
                    "--max-model-len", str(motion_config["qwen3"]["max_model_len"]),
                    *(["--skip-existing"] if args.resume else []),
                ),
                cwd=repo_root,
            )
            run_stage(
                python_stage(
                    scripts / "11_apply_motion_loudness_frame_diff_fallback.py",
                    "--qwen-metrics-csv", str(qwen_root / "qwen_cluster_visual_strength_metrics.csv"),
                    "--output-dir", str(final_root),
                ),
                cwd=repo_root,
            )
        except Exception as exc:
            # Impact Decay is already determined by the shared localized-event
            # metrics. Do not discard it when only the motion-only Qwen stage fails.
            motion_error = " ".join(str(exc).split())

    event_rows = read_csv(native_root / "metrics/source_mechanics_event_metrics.csv")
    events_by_id: dict[str, list[dict[str, str]]] = {}
    for row in event_rows:
        events_by_id.setdefault(row.get("video_id", ""), []).append(row)
    motion_metrics_path = final_root / "motion_loudness_metrics.csv"
    motion_metrics = {}
    if motion_metrics_path.is_file():
        motion_metrics = {
            row.get("generated_sample_id", ""): row for row in read_csv(motion_metrics_path)
        }

    output_rows = []
    for identifier, job in id_map.items():
        for evaluator in job["evaluators"]:
            if evaluator == "motion_loudness":
                metric = motion_metrics.get(identifier, {})
                passed = boolean(metric.get("frame_diff_fallback_postprocess_max_rms_db_no_margin"))
                pairs = [[1.0, 0.0]] if passed is True else ([[0.0, 1.0]] if passed is False else [])
                status = "error" if motion_error else ("success" if pairs else "invalid")
                reason = (
                    f"motion_loudness_backend_failed: {motion_error}"
                    if motion_error
                    else ("" if pairs else str(metric.get("qwen_visual_judgment_status", "no valid cluster judgment")))
                )
                output_rows.append(
                    {
                        "sample_id": job["sample_id"],
                        "evaluator": evaluator,
                        "status": status,
                        "reason": reason,
                        "evidence": {"level_pairs_db": pairs} if pairs else {},
                        "backend_version": "source-native-v2",
                        "artifacts": {"motion_metrics": str(motion_metrics_path)},
                    }
                )
            elif evaluator == "impact_decay":
                events = []
                for row in events_by_id.get(identifier, []):
                    fit = finite(row.get("decay_r2"))
                    residual = finite(row.get("tail_residual_mae_db"))
                    dynamic = finite(row.get("peak_to_floor_db"))
                    if fit is not None and residual is not None and dynamic is not None:
                        events.append(
                            {
                                "fit_r2": fit,
                                "tail_residual_mae_db": residual,
                                "peak_to_floor_db": dynamic,
                            }
                        )
                output_rows.append(
                    {
                        "sample_id": job["sample_id"],
                        "evaluator": evaluator,
                        "status": "success" if events else "invalid",
                        "reason": "" if events else "no valid impact-decay event",
                        "evidence": {"events": events} if events else {},
                        "backend_version": "source-native-v2",
                    }
                )
    write_jsonl(args.output, output_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
