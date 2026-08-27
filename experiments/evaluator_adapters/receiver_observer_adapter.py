"""Run and adapt the Range/Approach/Lateral receiver evaluator pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from common import (
    evaluator_inputs,
    ensure_symlink,
    finite,
    native_id,
    python_stage,
    read_csv,
    read_jsonl,
    run_stage,
    write_csv,
    write_jsonl,
)


SCORE_FIELDS = {
    "range_attenuation": (
        "sign_aware_windowed_inverse_square_fit_r2_proxy",
        "sign_aware_missing_reason",
    ),
    "approach_gain": (
        "approaching_consistency_score",
        "visual_approaching_enhancement_applicability_reason",
    ),
    "lateral_stability": (
        "loudness_stability_score",
        "visual_lateral_loudness_stability_applicability_reason",
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    work_dir = Path(args.work_dir).resolve()
    native_root = work_dir / "native"
    manifest_path = native_root / "receiver_manifest.csv"
    jobs = read_jsonl(args.input)
    id_map = {native_id(job["sample_id"]): job for job in jobs}
    native_rows = []
    early_rows = []
    for native_sample_id, job in id_map.items():
        inputs = evaluator_inputs(job, "receiver_observer")
        targets = inputs.get("detection_targets", inputs.get("candidate_detection_targets", ""))
        if isinstance(targets, list):
            targets = "|".join(map(str, targets))
        if not str(targets).strip():
            for evaluator in job["evaluators"]:
                early_rows.append(
                    {
                        "sample_id": job["sample_id"],
                        "evaluator": evaluator,
                        "status": "invalid",
                        "reason": "missing receiver_observer detection_targets",
                        "evidence": {},
                        "backend_version": "receiver-native-v2",
                    }
                )
            continue
        native_rows.append(
            {
                "sample_id": native_sample_id,
                "video_id": native_sample_id,
                "chunk_id": native_sample_id,
                "status": "ok",
                "skip_reason": "",
                "event_clip_path": job["video_path"],
                "video_path": job["video_path"],
                "event_audio_path": "",
                "candidate_detection_targets": str(targets),
                "detection_targets_key": str(inputs.get("detection_targets_key", "default")),
            }
        )
    write_csv(
        manifest_path,
        native_rows,
        (
            "sample_id",
            "video_id",
            "chunk_id",
            "status",
            "skip_reason",
            "event_clip_path",
            "video_path",
            "event_audio_path",
            "candidate_detection_targets",
            "detection_targets_key",
        ),
    )
    if not native_rows:
        write_jsonl(args.output, early_rows)
        return 0

    backend = repo_root / "experiments/evaluator_backends/receiver_observer"
    scripts = backend / "scripts"
    config_template = backend / "configs/receiver_observer_eval_config.yaml"
    unified_config = backend / "configs/receiver_observer_unified_v2_config.yaml"

    vda_repo = repo_root / "third_party/Video-Depth-Anything"
    gsam_repo = repo_root / "third_party/Grounded-Segment-Anything"
    ensure_symlink(
        repo_root / "checkpoints/video_depth_anything/metric_video_depth_anything_vitl.pth",
        vda_repo / "checkpoints/metric_video_depth_anything_vitl.pth",
    )
    config_data = yaml.safe_load(config_template.read_text(encoding="utf-8"))
    config_data["runtime"]["vda_repo"] = str(vda_repo)
    config_data["runtime"]["grounded_sam_repo"] = str(gsam_repo)
    config_data["gsam"]["config"] = str(
        gsam_repo / "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    )
    config_data["gsam"]["grounded_checkpoint"] = str(
        repo_root / "checkpoints/grounded_sam/groundingdino_swint_ogc.pth"
    )
    config_data["gsam"]["sam_checkpoint"] = str(
        repo_root / "checkpoints/grounded_sam/sam_vit_b_01ec64.pth"
    )
    config = native_root / "receiver_observer_eval_config.yaml"
    config.write_text(yaml.safe_dump(config_data, sort_keys=False), encoding="utf-8")
    depth_root = native_root / "depth"
    gsam_root = native_root / "gsam"
    audio_root = native_root / "audio_features"
    track_root = native_root / "tracks"
    metrics_root = native_root / "metrics"
    unified_csv = metrics_root / "receiver_observer_unified_v2_metrics.csv"
    screened_csv = metrics_root / "receiver_observer_screened_metrics.csv"
    applicability_csv = metrics_root / "receiver_observer_visual_applicability.csv"
    range_csv = metrics_root / "range_attenuation_sign_aware_metrics.csv"
    resume_flag = ["--skip-existing"] if args.resume else []

    stages = [
        python_stage(
            scripts / "01_run_vda_depth.py",
            "--config", str(config), "--manifest", str(manifest_path),
            "--output-root", str(depth_root), *resume_flag,
        ),
        python_stage(
            scripts / "02_run_grounded_sam_dynamic.py",
            "--config", str(config), "--manifest", str(manifest_path),
            "--output-root", str(gsam_root), "--depth-root", str(depth_root),
            "--dynamic-keyframes", *resume_flag,
        ),
        python_stage(
            scripts / "03_extract_audio_loudness.py",
            "--config", str(config), "--manifest", str(manifest_path),
            "--output-root", str(audio_root), *resume_flag,
        ),
        python_stage(
            scripts / "04_compute_depth_tracks.py",
            "--config", str(config), "--manifest", str(manifest_path),
            "--depth-root", str(depth_root), "--gsam-root", str(gsam_root),
            "--output-root", str(track_root), *resume_flag,
        ),
        python_stage(
            scripts / "05_compute_receiver_observer_metrics.py",
            "--config", str(config), "--manifest", str(manifest_path),
            "--audio-root", str(audio_root), "--track-root", str(track_root),
            "--output-root", str(metrics_root), *resume_flag,
        ),
        python_stage(
            scripts / "07_compute_receiver_observer_unified_v2.py",
            "--config", str(unified_config), "--manifest", str(manifest_path),
            "--audio_root", str(audio_root), "--track_root", str(track_root),
            "--split", "generated", "--out_csv", str(unified_csv),
        ),
        python_stage(
            scripts / "08_screen_receiver_observer_visual_applicability.py",
            "--config", str(unified_config),
            "--metrics_csv", str(unified_csv),
            "--track_root", str(track_root), "--out_csv", str(screened_csv),
            "--applicability_csv", str(applicability_csv),
            "--no-debug-plots",
        ),
        python_stage(
            scripts / "10_compute_sign_aware_range_attenuation.py",
            "--metrics-csv", str(unified_csv),
            "--audio-root", str(audio_root), "--track-root", str(track_root),
            "--out-csv", str(range_csv),
            "--summary-json", str(metrics_root / "range_attenuation_sign_aware_summary.json"),
        ),
    ]
    for stage in stages:
        run_stage(stage, cwd=repo_root)

    screened_metrics = {row["sample_id"]: row for row in read_csv(screened_csv)}
    range_metrics = {row["sample_id"]: row for row in read_csv(range_csv)}
    output_rows = list(early_rows)
    for native_sample_id, job in id_map.items():
        if not any(row["sample_id"] == native_sample_id for row in native_rows):
            continue
        for evaluator in job["evaluators"]:
            metric = (
                range_metrics.get(native_sample_id, {})
                if evaluator == "range_attenuation"
                else screened_metrics.get(native_sample_id, {})
            )
            score_field, reason_field = SCORE_FIELDS[evaluator]
            native_score = finite(metric.get(score_field))
            evidence = {}
            if native_score is not None:
                evidence = (
                    {"window_r2_values": [native_score]}
                    if evaluator == "range_attenuation"
                    else {"native_score": native_score}
                )
            output_rows.append(
                {
                    "sample_id": job["sample_id"],
                    "evaluator": evaluator,
                    "status": "success" if native_score is not None else "invalid",
                    "reason": "" if native_score is not None else str(
                        metric.get(reason_field, metric.get("skip_reason", "no valid receiver readout"))
                    ),
                    "evidence": evidence,
                    "backend_version": "receiver-native-v2",
                    "artifacts": {
                        "screened_metrics": str(screened_csv),
                        "range_metrics": str(range_csv),
                        "native_sample_id": native_sample_id,
                    },
                }
            )
    write_jsonl(args.output, output_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
