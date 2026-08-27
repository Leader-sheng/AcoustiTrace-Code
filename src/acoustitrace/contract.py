"""Validation helpers for the machine-readable paper contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_contract(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_contract(contract: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    prompts = contract["prompt_suite"]
    assignments = prompts["assignments"]
    expected_t2av = (
        assignments["range_attenuation"]
        + assignments["approach_gain_lateral_shared_pool"]
        + assignments["motion_loudness"]
        + assignments["impact_decay"]
        + assignments["rt60_consistency"]
        - prompts["range_receiver_overlap"]
    )
    if expected_t2av != prompts["unique"]["t2av"]:
        errors.append(f"T2AV prompt arithmetic gives {expected_t2av}")
    expected_i2av = expected_t2av + assignments["log_attack_time"]
    if expected_i2av != prompts["unique"]["i2av"]:
        errors.append(f"I2AV prompt arithmetic gives {expected_i2av}")
    splits = contract["rgbd_dataset"]["splits"]
    if sum(splits.values()) != contract["rgbd_dataset"]["total"]:
        errors.append("RGB-D split counts do not sum to the declared total")
    if contract["model_support"]["JavisDiT++"]["i2av"]:
        errors.append("JavisDiT++ I2AV must be unavailable")
    if contract["model_support"]["UniVerse-1"]["t2av"]:
        errors.append("UniVerse-1 T2AV must be unavailable")
    if contract["aggregation"]["invalid_score_policy"] != "exclude_from_conditional_mean":
        errors.append("invalid outputs must not be assigned zero")
    return errors

