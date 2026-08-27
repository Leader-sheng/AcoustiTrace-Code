"""Build the public 605 T2AV + 143 I2AV-LAT prompt manifests.

The T2AV source is the deduplicated current-605 score table: its 695 evaluator
assignment rows collapse to 605 unique prompts because the 90 receiver-observer
prompts belong to both Approach Gain and Lateral Stability.  The I2AV source is
the 143-row Log Attack Time generation manifest.  This script only converts
those frozen selections to the public runner schema; it does not resample them.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
from zipfile import ZipFile


DIMENSION_TO_EVALUATOR = {
    "distance_attenuation": "range_attenuation",
    "approaching_enhancement": "approach_gain",
    "loudness_stability": "lateral_stability",
    "action_loudness": "motion_loudness",
    "impact_decay": "impact_decay",
    "onset_causality": "causality_violation",
    "rt60_consistency": "rt60_consistency",
}

MEMBERSHIP_TO_CATEGORY = {
    ("range_attenuation",): "receiver_distance",
    ("approach_gain", "lateral_stability"): "receiver_observer",
    ("motion_loudness",): "source_motion_loudness",
    ("impact_decay",): "source_generation",
    ("causality_violation",): "time_causality",
    ("rt60_consistency",): "propagation_rt60_500_consistency",
}

MEMBERSHIP_ORDER = {
    membership: index for index, membership in enumerate(MEMBERSHIP_TO_CATEGORY)
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def first_text(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = str(row.get(key, "")).strip()
        if value and value.lower() != "nan":
            return value
    return ""


def finite_float(row: dict[str, Any], key: str) -> float | None:
    try:
        value = float(row.get(key))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def detection_targets(metadata: dict[str, Any]) -> list[str]:
    targets: list[str] = []
    for key in ("source_object", "source_type"):
        value = first_text(metadata, key).replace("_", " ")
        if value and value not in targets:
            targets.append(value)
    return targets


def build_t2av(
    scores_path: Path,
    master_pool_path: Path,
    range_pool_path: Path,
) -> list[dict[str, Any]]:
    score_rows = read_csv(scores_path)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in score_rows:
        sample_id = first_text(row, "sample_id")
        dimension = first_text(row, "dimension")
        if not sample_id or dimension not in DIMENSION_TO_EVALUATOR:
            raise ValueError(f"invalid current-605 score row: {row}")
        grouped[sample_id].append(row)

    master_by_id = {
        first_text(row, "sample_id"): row for row in read_jsonl(master_pool_path)
    }
    range_by_id = {
        first_text(row, "new_sample_id"): row for row in read_jsonl(range_pool_path)
    }

    records: list[dict[str, Any]] = []
    for sample_id, rows in grouped.items():
        membership = tuple(
            evaluator
            for evaluator in DIMENSION_TO_EVALUATOR.values()
            if evaluator
            in {DIMENSION_TO_EVALUATOR[first_text(row, "dimension")] for row in rows}
        )
        if membership not in MEMBERSHIP_TO_CATEGORY:
            raise ValueError(f"unsupported membership for {sample_id}: {membership}")
        prompt_text = first_text(rows[0], "text_prompt", "prompt_text", "source_prompt")
        if not prompt_text:
            raise ValueError(f"missing prompt text for {sample_id}")

        evaluator_inputs: dict[str, Any] = {}
        if set(membership) & {"range_attenuation", "approach_gain", "lateral_stability"}:
            metadata = master_by_id.get(sample_id) or range_by_id.get(sample_id) or {}
            targets = detection_targets(metadata)
            if not targets:
                raise ValueError(f"missing receiver detection targets for {sample_id}")
            evaluator_inputs["receiver_observer"] = {"detection_targets": targets}

        records.append(
            {
                "prompt_id": sample_id,
                "task": "t2av",
                "prompt_text": prompt_text,
                "category": MEMBERSHIP_TO_CATEGORY[membership],
                "evaluator_membership": list(membership),
                "conditioning_asset_id": "",
                "conditioning_asset_path": "",
                "evaluator_inputs": evaluator_inputs,
                "split": "benchmark",
            }
        )

    records.sort(
        key=lambda row: (
            MEMBERSHIP_ORDER[tuple(row["evaluator_membership"])], row["prompt_id"]
        )
    )
    if len(records) != 605:
        raise ValueError(f"T2AV manifest has {len(records)} rows, expected 605")
    return records


def conditioning_assets(archive_path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with ZipFile(archive_path) as archive:
        for name in archive.namelist():
            basename = PurePosixPath(name).name
            marker = "__source_log_attack_time__"
            if marker not in basename or not basename.lower().endswith(".png"):
                continue
            prompt_id = basename.split(marker, maxsplit=1)[1][:-4]
            if prompt_id in result:
                raise ValueError(f"duplicate conditioning asset for {prompt_id}")
            result[prompt_id] = basename
    return result


def build_i2av(
    source_path: Path, archive_path: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    assets = conditioning_assets(archive_path)
    source_rows = read_jsonl(source_path)
    records: list[dict[str, Any]] = []
    reference_sources: list[dict[str, Any]] = []
    for row in source_rows:
        prompt_id = first_text(row, "prompt_id", "i2av_global_id")
        sample_id = first_text(row, "sample_id", "material_video_id")
        prompt_text = first_text(
            row,
            "final_revised_prompt",
            "rewritten_generation_prompt",
            "generation_prompt",
        )
        if not prompt_id or not sample_id or not prompt_text:
            raise ValueError(f"invalid I2AV-LAT source row: {row}")
        basename = assets.get(prompt_id)
        if not basename:
            raise ValueError(f"missing conditioning asset for {prompt_id}")
        records.append(
            {
                "prompt_id": prompt_id,
                "task": "i2av",
                "prompt_text": prompt_text,
                "category": "source_log_attack_time",
                "evaluator_membership": ["log_attack_time"],
                "conditioning_asset_id": prompt_id,
                "conditioning_asset_path": f"../i2av_conditioning_assets/{basename}",
                "evaluator_inputs": {
                    "log_attack_time": {
                        "reference_audio_path": f"data/references/i2av_lat/{prompt_id}.wav"
                    }
                },
                "split": "benchmark",
            }
        )
        reference_sources.append(
            {
                "prompt_id": prompt_id,
                "dataset": "Greatest Hits",
                "source_sample_id": sample_id,
                "source_video_filename": f"{sample_id}.mp4",
                "conditioning_asset_path": f"data/i2av_conditioning_assets/{basename}",
                "reference_audio_path": f"data/references/i2av_lat/{prompt_id}.wav",
                "target_clip_duration_sec": 5.0,
                "selected_event_start_sec": finite_float(
                    row, "selected_event_start_sec"
                ),
                "selected_event_end_sec": finite_float(row, "selected_event_end_sec"),
                "source_gt_clip_duration_sec": finite_float(row, "gt_video_duration"),
            }
        )
    records.sort(key=lambda row: row["prompt_id"])
    reference_sources.sort(key=lambda row: row["prompt_id"])
    if len(records) != 143 or len({row["prompt_id"] for row in records}) != 143:
        raise ValueError("I2AV-LAT manifest must contain 143 unique prompts")
    if set(assets) != {row["prompt_id"] for row in records}:
        raise ValueError("I2AV-LAT source rows and conditioning archive do not match")
    if len({row["source_sample_id"] for row in reference_sources}) != 143:
        raise ValueError("I2AV-LAT source mapping must contain 143 unique videos")
    return records, reference_sources


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t2av-scores", type=Path, required=True)
    parser.add_argument("--master-pool", type=Path, required=True)
    parser.add_argument("--range-pool", type=Path, required=True)
    parser.add_argument("--i2av-source", type=Path, required=True)
    parser.add_argument("--conditioning-archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    t2av = build_t2av(args.t2av_scores, args.master_pool, args.range_pool)
    i2av, i2av_reference_sources = build_i2av(
        args.i2av_source, args.conditioning_archive
    )
    write_jsonl(args.output_dir / "t2av_605.jsonl", t2av)
    write_jsonl(args.output_dir / "i2av_lat_143.jsonl", i2av)
    write_jsonl(
        args.output_dir.parent / "references" / "i2av_lat_sources.jsonl",
        i2av_reference_sources,
    )
    print(f"wrote {len(t2av)} T2AV prompts and {len(i2av)} I2AV-LAT prompts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
