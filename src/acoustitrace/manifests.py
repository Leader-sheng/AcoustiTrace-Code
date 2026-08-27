"""Validation helpers for frozen prompt and generated-output manifests."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping


EVALUATORS = (
    "range_attenuation",
    "approach_gain",
    "lateral_stability",
    "motion_loudness",
    "impact_decay",
    "causality_violation",
    "log_attack_time",
    "rt60_consistency",
)


@dataclass(frozen=True)
class PromptRecord:
    prompt_id: str
    task: str
    prompt_text: str
    evaluator_membership: tuple[str, ...]
    conditioning_asset_id: str = ""
    conditioning_asset_path: str = ""
    split: str = "benchmark"
    category: str = ""
    evaluator_inputs: Mapping[str, Any] = field(default_factory=dict)
    manifest_path: str = ""


def _read_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    if suffix in {".jsonl", ".ndjson"}:
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise ValueError(f"line {line_number} is not a JSON object")
                rows.append(item)
        return rows
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(data, dict) and isinstance(data.get("prompts"), list):
            data = data["prompts"]
        if not isinstance(data, list) or not all(isinstance(row, dict) for row in data):
            raise ValueError("JSON manifest must be a list of objects or {'prompts': [...]}")
        return data
    raise ValueError(f"unsupported manifest format: {path.suffix}")


def _members(value: Any) -> tuple[str, ...]:
    if isinstance(value, list):
        items = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            items = []
        elif text.startswith("["):
            items = json.loads(text)
        else:
            items = text.replace(",", ";").split(";")
    else:
        items = []
    return tuple(sorted({str(item).strip().lower() for item in items if str(item).strip()}))


def _json_mapping(value: Any, *, field_name: str, row_number: int) -> dict[str, Any]:
    if value in (None, ""):
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"row {row_number} has invalid JSON in {field_name}: {exc}"
            ) from exc
        if isinstance(parsed, Mapping):
            return dict(parsed)
    raise ValueError(f"row {row_number} field {field_name} must be a JSON object")


def load_prompt_manifest(path: str | Path) -> list[PromptRecord]:
    source = Path(path).resolve()
    records: list[PromptRecord] = []
    for index, row in enumerate(_read_records(source), start=1):
        try:
            records.append(
                PromptRecord(
                    prompt_id=str(row["prompt_id"]).strip(),
                    task=str(row["task"]).strip().lower(),
                    prompt_text=str(row["prompt_text"]).strip(),
                    evaluator_membership=_members(row.get("evaluator_membership", [])),
                    conditioning_asset_id=str(row.get("conditioning_asset_id", "")).strip(),
                    conditioning_asset_path=str(row.get("conditioning_asset_path", "")).strip(),
                    split=str(row.get("split", "benchmark")).strip() or "benchmark",
                    category=str(row.get("category", row.get("prompt_category", ""))).strip(),
                    evaluator_inputs=_json_mapping(
                        row.get("evaluator_inputs", {}),
                        field_name="evaluator_inputs",
                        row_number=index,
                    ),
                    manifest_path=str(source),
                )
            )
        except KeyError as exc:
            raise ValueError(f"row {index} is missing required field {exc.args[0]!r}") from exc
    return records


def _duplicates(values: Iterable[str]) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return duplicates


def validate_prompt_manifest(
    records: list[PromptRecord],
    contract: dict[str, Any],
    *,
    allow_subset: bool = False,
) -> list[str]:
    errors: list[str] = []
    if not records:
        return ["manifest contains no prompt rows"]

    duplicate_ids = _duplicates(f"{row.task}:{row.prompt_id}" for row in records)
    if duplicate_ids:
        errors.append(f"duplicate task/prompt IDs: {', '.join(sorted(duplicate_ids)[:10])}")

    unknown_tasks = sorted({row.task for row in records} - {"t2av", "i2av"})
    if unknown_tasks:
        errors.append(f"unknown tasks: {', '.join(unknown_tasks)}")

    for row in records:
        if not row.prompt_id:
            errors.append("a row has an empty prompt_id")
        if not row.prompt_text:
            errors.append(f"{row.task}:{row.prompt_id} has empty prompt_text")
        if not row.evaluator_membership:
            errors.append(f"{row.task}:{row.prompt_id} has no evaluator membership")
        unknown = sorted(set(row.evaluator_membership) - set(EVALUATORS))
        if unknown:
            errors.append(
                f"{row.task}:{row.prompt_id} has unknown evaluators: {', '.join(unknown)}"
            )
        if row.task == "i2av" and not (
            row.conditioning_asset_id or row.conditioning_asset_path
        ):
            errors.append(f"i2av:{row.prompt_id} has no conditioning asset")
        if row.task == "t2av" and "log_attack_time" in row.evaluator_membership:
            errors.append(f"t2av:{row.prompt_id} incorrectly includes log_attack_time")

    expected_unique = contract["prompt_suite"]["unique"]
    assignments = contract["prompt_suite"]["assignments"]
    expected_assignment = {
        "range_attenuation": assignments["range_attenuation"],
        "approach_gain": assignments["approach_gain"],
        "lateral_stability": assignments["lateral_stability"],
        "motion_loudness": assignments["motion_loudness"],
        "impact_decay": assignments["impact_decay"],
        "causality_violation": assignments["causality_violation"],
        "rt60_consistency": assignments["rt60_consistency"],
    }

    for task in ("t2av", "i2av"):
        task_rows = [row for row in records if row.task == task]
        if not task_rows:
            continue
        if not allow_subset and len(task_rows) != expected_unique[task]:
            errors.append(
                f"{task} has {len(task_rows)} unique prompts; expected {expected_unique[task]}"
            )

        sets = {
            evaluator: {
                row.prompt_id for row in task_rows if evaluator in row.evaluator_membership
            }
            for evaluator in EVALUATORS
        }
        if not allow_subset:
            for evaluator, expected in expected_assignment.items():
                actual = len(sets[evaluator])
                if actual != expected:
                    errors.append(
                        f"{task}:{evaluator} has {actual} assignments; expected {expected}"
                    )
            expected_lat = assignments["log_attack_time"] if task == "i2av" else 0
            if len(sets["log_attack_time"]) != expected_lat:
                errors.append(
                    f"{task}:log_attack_time has {len(sets['log_attack_time'])} assignments; "
                    f"expected {expected_lat}"
                )

        if sets["approach_gain"] != sets["lateral_stability"]:
            errors.append(f"{task}: Approach Gain and Lateral Stability pools differ")
        overlap = sets["range_attenuation"] & sets["approach_gain"]
        expected_overlap = contract["prompt_suite"]["range_receiver_overlap"]
        if not allow_subset and len(overlap) != expected_overlap:
            errors.append(
                f"{task}: Range/receiver overlap is {len(overlap)}; expected {expected_overlap}"
            )
        causality_sources = sets["motion_loudness"] | sets["impact_decay"]
        if not sets["causality_violation"].issubset(causality_sources):
            errors.append(
                f"{task}: Causality contains prompts outside Motion--Loudness/Impact pools"
            )
        if task == "i2av":
            non_lat = set().union(
                *(sets[name] for name in EVALUATORS if name != "log_attack_time")
            )
            if sets["log_attack_time"] & non_lat:
                errors.append("i2av: Log Attack Time prompts are not disjoint from core prompts")

    return list(dict.fromkeys(errors))


def load_output_manifest(path: str | Path) -> list[dict[str, Any]]:
    return _read_records(Path(path))


def validate_output_manifest(
    outputs: list[dict[str, Any]],
    prompts: list[PromptRecord],
    *,
    check_files: bool = False,
) -> list[str]:
    errors: list[str] = []
    expected = {(row.task, row.prompt_id) for row in prompts}
    observed: list[tuple[str, str]] = []
    for index, row in enumerate(outputs, start=1):
        task = str(row.get("task", "")).strip().lower()
        prompt_id = str(row.get("prompt_id", row.get("sample_id", ""))).strip()
        if not task or not prompt_id:
            errors.append(f"output row {index} is missing task or prompt_id")
            continue
        observed.append((task, prompt_id))
        status = str(row.get("status", "success")).strip().lower() or "success"
        video_path = str(row.get("video_path", "")).strip()
        if status == "success" and not video_path:
            errors.append(f"{task}:{prompt_id} is successful but has no video_path")
        if check_files and status == "success" and video_path and not Path(video_path).is_file():
            errors.append(f"{task}:{prompt_id} video does not exist: {video_path}")

    duplicates = _duplicates(f"{task}:{prompt_id}" for task, prompt_id in observed)
    if duplicates:
        errors.append(f"duplicate output rows: {', '.join(sorted(duplicates)[:10])}")
    observed_set = set(observed)
    missing = sorted(expected - observed_set)
    extra = sorted(observed_set - expected)
    if missing:
        errors.append(f"missing {len(missing)} prompt outputs; first: {missing[:5]}")
    if extra:
        errors.append(f"found {len(extra)} unknown prompt outputs; first: {extra[:5]}")
    return errors


def generation_request(record: PromptRecord) -> dict[str, Any]:
    if Path(record.prompt_id).name != record.prompt_id or record.prompt_id in {".", ".."}:
        raise ValueError(f"prompt_id {record.prompt_id!r} is not safe as an MP4 filename")
    output_subdir = "i2av_lat" if record.task == "i2av" else record.task
    output_filename = f"{record.prompt_id}.mp4"
    request: dict[str, Any] = {
        "prompt_id": record.prompt_id,
        "task": record.task,
        "prompt": record.prompt_text,
        "evaluator_membership": list(record.evaluator_membership),
        "output_subdir": output_subdir,
        "output_filename": output_filename,
        "output_relpath": f"{output_subdir}/{output_filename}",
    }
    if record.conditioning_asset_id:
        request["conditioning_asset_id"] = record.conditioning_asset_id
    if record.conditioning_asset_path:
        request["conditioning_asset_path"] = record.conditioning_asset_path
    return request
