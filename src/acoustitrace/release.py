"""Validation for the first public 605 T2AV + 143 I2AV-LAT suite."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from collections.abc import Mapping

from .manifests import EVALUATORS, PromptRecord


def default_release_profile_path() -> Path:
    return Path(__file__).with_name("release_profile.json")


def load_release_profile(path: str | Path | None = None) -> dict[str, Any]:
    source = Path(path) if path else default_release_profile_path()
    data = json.loads(source.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or not isinstance(data.get("tasks"), dict):
        raise ValueError(f"unsupported release profile: {source}")
    return data


def _membership_key(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    return tuple(sorted(str(value).strip().lower() for value in values))


def _evaluator_inputs(row: PromptRecord, group: str) -> dict[str, Any]:
    raw = row.evaluator_inputs
    if not isinstance(raw, Mapping):
        return {}
    nested = raw.get(group, {})
    return dict(raw) | (dict(nested) if isinstance(nested, Mapping) else {})


def validate_release_suite(
    records: list[PromptRecord],
    profile: dict[str, Any],
    *,
    allow_subset: bool = False,
) -> list[str]:
    """Validate the public rebalanced suite without changing the paper contract."""

    errors: list[str] = []
    if not records:
        return ["release suite contains no prompts"]

    seen: set[tuple[str, str]] = set()
    for row in records:
        key = (row.task, row.prompt_id)
        if key in seen:
            errors.append(f"duplicate release prompt: {row.task}:{row.prompt_id}")
        seen.add(key)
        if not row.prompt_id:
            errors.append("a release prompt has an empty prompt_id")
        if not row.prompt_text:
            errors.append(f"{row.task}:{row.prompt_id} has empty prompt_text")
        if row.task not in profile["tasks"]:
            errors.append(f"{row.task}:{row.prompt_id} has unsupported task {row.task!r}")
            continue
        unknown = sorted(set(row.evaluator_membership) - set(EVALUATORS))
        if unknown:
            errors.append(
                f"{row.task}:{row.prompt_id} has unknown evaluators: {', '.join(unknown)}"
            )
        allowed = {
            _membership_key(values)
            for values in profile["tasks"][row.task]["allowed_memberships"]
        }
        if _membership_key(row.evaluator_membership) not in allowed:
            errors.append(
                f"{row.task}:{row.prompt_id} has unsupported release membership "
                f"{list(row.evaluator_membership)!r}"
            )
        if row.task == "i2av" and not (
            row.conditioning_asset_id or row.conditioning_asset_path
        ):
            errors.append(f"i2av:{row.prompt_id} has no conditioning asset")
        if {"range_attenuation", "approach_gain", "lateral_stability"} & set(
            row.evaluator_membership
        ):
            receiver_inputs = _evaluator_inputs(row, "receiver_observer")
            targets = receiver_inputs.get(
                "detection_targets", receiver_inputs.get("candidate_detection_targets")
            )
            if not targets:
                errors.append(
                    f"{row.task}:{row.prompt_id} has no receiver detection_targets"
                )
        if "log_attack_time" in row.evaluator_membership:
            lat_inputs = _evaluator_inputs(row, "log_attack_time")
            if not str(lat_inputs.get("reference_audio_path", "")).strip():
                errors.append(
                    f"{row.task}:{row.prompt_id} has no Log Attack Time reference_audio_path"
                )

    if allow_subset:
        return list(dict.fromkeys(errors))

    for task, task_profile in profile["tasks"].items():
        task_rows = [row for row in records if row.task == task]
        expected_unique = int(task_profile["unique_prompts"])
        if len(task_rows) != expected_unique:
            errors.append(
                f"{task} has {len(task_rows)} prompts; expected {expected_unique}"
            )
        for evaluator, expected in task_profile["assignment_counts"].items():
            actual = sum(evaluator in row.evaluator_membership for row in task_rows)
            if actual != int(expected):
                errors.append(
                    f"{task}:{evaluator} has {actual} assignments; expected {expected}"
                )

    return list(dict.fromkeys(errors))
