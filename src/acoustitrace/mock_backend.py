"""Protocol-only backend used by tests; never use it for benchmark results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MOCK_EVIDENCE = {
    "range_attenuation": {"window_r2_values": [0.8, 0.9]},
    "approach_gain": {"native_score": 0.75},
    "lateral_stability": {"native_score": 0.7},
    "motion_loudness": {"level_pairs_db": [[-10.0, -20.0], [-12.0, -18.0]]},
    "impact_decay": {
        "fit_r2": 0.8,
        "tail_residual_mae_db": 1.0,
        "peak_to_floor_db": 20.0,
    },
    "causality_violation": {"onset_delays_seconds": [0.02, 0.04]},
    "rt60_consistency": {"audio_rt60": 1.0, "visual_rt60": 1.0},
    "log_attack_time": {
        "generated_attack_seconds": 0.02,
        "reference_attack_seconds": 0.02,
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with Path(args.input).open("r", encoding="utf-8") as source, target.open(
        "w", encoding="utf-8"
    ) as destination:
        for line in source:
            if not line.strip():
                continue
            job = json.loads(line)
            for evaluator in job["evaluators"]:
                destination.write(
                    json.dumps(
                        {
                            "sample_id": job["sample_id"],
                            "evaluator": evaluator,
                            "status": "success",
                            "evidence": MOCK_EVIDENCE[evaluator],
                            "backend_version": "protocol-mock-v1",
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
