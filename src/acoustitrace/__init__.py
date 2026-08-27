"""Paper-aligned AcoustiTrace scoring utilities."""

from .aggregate import ScoreRecord, bootstrap_cell, matched_valid
from .guidance import decoded_mel_pairwise_loss
from .scores import (
    ScoreResult,
    causality_violation,
    impact_decay,
    log_attack_time,
    motion_loudness,
    native_receiver_score,
    range_attenuation,
    range_reference_db,
    rt60_consistency,
)

__all__ = [
    "ScoreRecord",
    "ScoreResult",
    "bootstrap_cell",
    "causality_violation",
    "decoded_mel_pairwise_loss",
    "impact_decay",
    "log_attack_time",
    "matched_valid",
    "motion_loudness",
    "native_receiver_score",
    "range_attenuation",
    "range_reference_db",
    "rt60_consistency",
]
