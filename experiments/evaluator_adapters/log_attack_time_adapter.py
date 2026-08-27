"""Adapt the native Log Attack Time evaluator to the public JSONL protocol."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from common import evaluator_inputs, python_stage, read_csv, read_jsonl, run_stage, write_csv, write_jsonl


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
    pairs_path = work_dir / "lat_pairs.csv"
    native_output = work_dir / "lat_native.csv"
    jobs = read_jsonl(args.input)
    pairs = []
    invalid = []
    for job in jobs:
        inputs = evaluator_inputs(job, "log_attack_time")
        reference = str(inputs.get("reference_audio_path", ""))
        if not reference:
            invalid.append(
                {
                    "sample_id": job["sample_id"],
                    "evaluator": "log_attack_time",
                    "status": "invalid",
                    "reason": "missing evaluator_inputs.log_attack_time.reference_audio_path",
                    "evidence": {},
                    "backend_version": "lat-native-v1",
                }
            )
            continue
        pairs.append(
            {
                "sample_id": job["sample_id"],
                "generated_path": job["video_path"],
                "reference_path": reference,
                "generated_onset_sec": inputs.get("generated_onset_sec", ""),
                "reference_onset_sec": inputs.get("reference_onset_sec", ""),
            }
        )
    write_csv(
        pairs_path,
        pairs,
        (
            "sample_id",
            "generated_path",
            "reference_path",
            "generated_onset_sec",
            "reference_onset_sec",
        ),
    )
    output_rows = list(invalid)
    if pairs:
        native_script = (
            repo_root
            / "experiments/evaluator_backends/log_attack_time/log_attack_time.py"
        )
        if not (args.resume and native_output.is_file()):
            run_stage(
                python_stage(
                    native_script,
                    "--pairs",
                    str(pairs_path),
                    "--output",
                    str(native_output),
                ),
                cwd=repo_root,
            )
        for row in read_csv(native_output):
            valid = str(row.get("valid", "")).lower() in {"true", "1", "yes"}
            output_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "evaluator": "log_attack_time",
                    "status": "success" if valid else "invalid",
                    "reason": row.get("error", ""),
                    "evidence": {
                        "generated_attack_seconds": row.get("generated_attack_sec", ""),
                        "reference_attack_seconds": row.get("reference_attack_sec", ""),
                    }
                    if valid
                    else {},
                    "backend_version": "lat-native-v1",
                    "artifacts": {"native_csv": str(native_output)},
                }
            )
    write_jsonl(args.output, output_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
