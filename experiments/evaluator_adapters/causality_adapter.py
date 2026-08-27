"""Run event localization/clustering and adapt Causality Violation evidence."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import yaml

from common import finite, native_id, python_stage, read_csv, read_jsonl, run_stage, write_csv, write_jsonl


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
    jobs = read_jsonl(args.input)
    id_map = {native_id(job["sample_id"]): job for job in jobs}

    # Causality uses the same OV-AVEL/FlexSED event localization as the source
    # evaluator. Run it in a private cache so the public top-level runner does
    # not depend on cross-group prompt overlap.
    source_input = work_dir / "source_input.jsonl"
    source_output = work_dir / "source_placeholder_evidence.jsonl"
    source_jobs = [{**job, "evaluators": ["impact_decay"]} for job in jobs]
    write_jsonl(source_input, source_jobs)
    source_adapter = repo_root / "experiments/evaluator_adapters/source_mechanics_adapter.py"
    source_work = work_dir / "source_preprocess"
    source_command = python_stage(
        source_adapter,
        "--input", str(source_input),
        "--output", str(source_output),
        "--work-dir", str(source_work),
        "--repo-root", str(repo_root),
        *(["--resume"] if args.resume else []),
    )
    run_stage(source_command, cwd=repo_root)

    source_native = source_work / "native"
    project_root = work_dir / "causality_project"
    (project_root / "outputs/ov_avel").mkdir(parents=True, exist_ok=True)
    (project_root / "outputs/flexsed").mkdir(parents=True, exist_ok=True)
    (project_root / "time_causality_eval/configs").mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        source_native / "ov_avel/visual_events.csv",
        project_root / "outputs/ov_avel/visual_events.csv",
    )
    shutil.copy2(
        source_native / "flexsed/audio_events.csv",
        project_root / "outputs/flexsed/audio_events.csv",
    )
    causality_backend = repo_root / "experiments/evaluator_backends/time_causality_eval"
    shutil.copy2(
        causality_backend / "configs/event_label_mapping.yaml",
        project_root / "time_causality_eval/configs/event_label_mapping.yaml",
    )

    output_root = work_dir / "native"
    manifest_rows = [
        {
            "sample_id": identifier,
            "video_id": identifier,
            "chunk_id": identifier,
            "video_path": job["video_path"],
            "status": "ok",
            "skip_reason": "",
        }
        for identifier, job in id_map.items()
    ]
    write_csv(
        output_root / "manifests/time_causality_manifest.csv",
        manifest_rows,
        ("sample_id", "video_id", "chunk_id", "video_path", "status", "skip_reason"),
    )
    config = yaml.safe_load(
        (causality_backend / "configs/time_causality_config.yaml").read_text(encoding="utf-8")
    )
    config["input"]["project_root"] = str(project_root)
    config["output"]["output_root"] = str(output_root)
    config_path = work_dir / "time_causality_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    scripts = causality_backend / "scripts"
    common_args = [
        "--config", str(config_path),
        "--project-root", str(project_root),
        "--output-root", str(output_root),
    ]
    if args.resume:
        common_args.append("--skip-existing")
    for name in (
        "02_parse_visual_audio_events.py",
        "02b_cluster_events.py",
        "03_match_events_for_causality.py",
        "04_compute_time_causality_metrics.py",
    ):
        run_stage(python_stage(scripts / name, *common_args), cwd=repo_root)

    events_path = output_root / "metrics/time_causality_event_metrics.csv"
    delays_by_id: dict[str, list[float]] = {}
    if events_path.is_file():
        for row in read_csv(events_path):
            delay = finite(row.get("delay_sec"))
            if delay is not None:
                delays_by_id.setdefault(row.get("sample_id", ""), []).append(delay)
    output_rows = []
    for identifier, job in id_map.items():
        delays = delays_by_id.get(identifier, [])
        output_rows.append(
            {
                "sample_id": job["sample_id"],
                "evaluator": "causality_violation",
                "status": "success" if delays else "invalid",
                "reason": "" if delays else "no reliable matched audio-visual event",
                "evidence": {"onset_delays_seconds": delays} if delays else {},
                "backend_version": "causality-native-v1",
                "artifacts": {"event_metrics": str(events_path)},
            }
        )
    write_jsonl(args.output, output_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
