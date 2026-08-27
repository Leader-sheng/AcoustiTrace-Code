#!/usr/bin/env python3
"""Cross-check release tables against the final paper contract."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def read_csv(name: str) -> list[dict[str, str]]:
    with (ROOT / "data" / "paper" / name).open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        return list(csv.DictReader(handle))


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def main() -> int:
    sys.path.insert(0, str(ROOT / "src"))
    from acoustitrace.contract import validate_contract

    errors: list[str] = []
    contract = json.loads(
        (ROOT / "configs" / "paper_contract.json").read_text(encoding="utf-8")
    )
    errors.extend(validate_contract(contract))
    packaged_contract = json.loads(
        (ROOT / "src" / "acoustitrace" / "paper_contract.json").read_text(
            encoding="utf-8"
        )
    )
    if packaged_contract != contract:
        fail(errors, "packaged and source-tree paper contracts differ")

    validity = read_csv("validity_counts.csv")
    results = read_csv("conditional_means_ci.csv")
    if len(validity) != 135:
        fail(errors, f"expected 135 validity rows, found {len(validity)}")
    if len(results) != 135:
        fail(errors, f"expected 135 result rows, found {len(results)}")

    def key(row: dict[str, str]) -> tuple[str, str, str]:
        return row["task"], row["model"], row["evaluator"]

    validity_map = {key(row): row for row in validity}
    result_map = {key(row): row for row in results}
    if len(validity_map) != len(validity):
        fail(errors, "duplicate validity cell")
    if len(result_map) != len(results):
        fail(errors, "duplicate result cell")
    if set(validity_map) != set(result_map):
        fail(errors, "validity and result tables contain different cells")

    assignment = contract["prompt_suite"]["assignments"]
    totals = {
        "range_attenuation": assignment["range_attenuation"],
        "approach_gain": assignment["approach_gain"],
        "lateral_stability": assignment["lateral_stability"],
        "motion_loudness": assignment["motion_loudness"],
        "impact_decay": assignment["impact_decay"],
        "causality_violation": assignment["causality_violation"],
        "log_attack_time": assignment["log_attack_time"],
        "rt60_consistency": assignment["rt60_consistency"],
    }
    attempts = {"t2av": 0, "i2av": 0}
    invalid = {"t2av": 0, "i2av": 0}
    for cell, row in validity_map.items():
        task, model, evaluator = cell
        supported = contract["model_support"][model][task]
        expected_status = "available" if supported else "not_available"
        if row["status"] != expected_status:
            fail(errors, f"support mismatch for {cell}")
        n_valid = int(row["n_valid"])
        n_total = int(row["n_total"])
        if supported and n_total != totals[evaluator]:
            fail(errors, f"wrong n_total for {cell}: {n_total}")
        if not 0 <= n_valid <= n_total:
            fail(errors, f"invalid coverage counts for {cell}")
        attempts[task] += n_total
        invalid[task] += n_total - n_valid

        result = result_map[cell]
        if result["status"] != row["status"]:
            fail(errors, f"status mismatch for {cell}")
        if supported:
            mean = float(result["mean"])
            low = float(result["ci_low"])
            high = float(result["ci_high"])
            if not (0.0 <= low <= mean <= high <= 100.0):
                fail(errors, f"invalid mean/CI ordering for {cell}")
        elif any(result[field] for field in ("mean", "ci_low", "ci_high")):
            fail(errors, f"unsupported cell has result values: {cell}")

    if attempts != {"t2av": 7768, "i2av": 8912}:
        fail(errors, f"attempt totals differ from supplement: {attempts}")
    if invalid != {"t2av": 649, "i2av": 993}:
        fail(errors, f"invalid totals differ from supplement: {invalid}")

    matched = read_csv("matched_valid.csv")
    if len(matched) != 45:
        fail(errors, f"expected 45 matched-valid model rows, found {len(matched)}")
    if len(read_csv("human_agreement.csv")) != 8:
        fail(errors, "human agreement table must contain eight evaluators")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("OK: paper contract and release tables are internally consistent")
    print("OK: 135 cells; 7,768 T2AV and 8,912 I2AV attempts")
    print("OK: invalid counts are 649 T2AV and 993 I2AV")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
