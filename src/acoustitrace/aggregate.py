"""Validity-aware aggregation used by AcoustiTrace."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from statistics import fmean
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class ScoreRecord:
    sample_id: str
    model: str
    task: str
    evaluator: str
    valid: bool
    score: float | None

    def __post_init__(self) -> None:
        if self.valid:
            if self.score is None or not math.isfinite(float(self.score)):
                raise ValueError("valid records require a finite score")
            if not 0.0 <= float(self.score) <= 100.0:
                raise ValueError("scores must lie on [0, 100]")


def percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute a percentile of an empty sequence")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def bootstrap_cell(
    records: Iterable[ScoreRecord],
    *,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> dict[str, float | int]:
    rows = list(records)
    valid_scores = [float(row.score) for row in rows if row.valid and row.score is not None]
    if not rows:
        raise ValueError("cell contains no attempted outputs")
    if not valid_scores:
        raise ValueError("cell contains no valid outputs")
    rng = random.Random(seed)
    size = len(valid_scores)
    boot = [
        fmean(valid_scores[rng.randrange(size)] for _ in range(size))
        for _ in range(n_resamples)
    ]
    boot.sort()
    return {
        "n_total": len(rows),
        "n_valid": size,
        "valid_rate": size / len(rows),
        "mean": fmean(valid_scores),
        "ci_low": percentile(boot, 0.025),
        "ci_high": percentile(boot, 0.975),
        "n_resamples": n_resamples,
        "seed": seed,
    }


def matched_valid(
    records: Iterable[ScoreRecord],
    models: Sequence[str],
) -> Mapping[str, list[ScoreRecord]]:
    """Return records on prompt IDs valid for every requested model."""

    requested = set(models)
    by_sample: dict[str, dict[str, ScoreRecord]] = {}
    for row in records:
        if row.model in requested:
            by_sample.setdefault(row.sample_id, {})[row.model] = row
    common_ids = {
        sample_id
        for sample_id, model_rows in by_sample.items()
        if set(model_rows) == requested and all(row.valid for row in model_rows.values())
    }
    return {
        model: [by_sample[sample_id][model] for sample_id in sorted(common_ids)]
        for model in models
    }

