"""Final score mappings specified by the AAAI 2027 paper and supplement.

Evidence extraction is intentionally separate. Detector, tracker, depth, and
audio-feature backends should emit finite native measurements before calling
these functions.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence


@dataclass(frozen=True)
class ScoreResult:
    valid: bool
    score: float | None
    reason: str | None = None


def _invalid(reason: str) -> ScoreResult:
    return ScoreResult(valid=False, score=None, reason=reason)


def _finite(value: float) -> bool:
    return math.isfinite(float(value))


def _clip(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def motion_loudness(level_pairs_db: Iterable[Sequence[float]]) -> ScoreResult:
    """Score visually stronger/visually weaker event-level RMS pairs.

    Each pair is `(stronger_visible_event_db, weaker_visible_event_db)`. The
    final no-margin score is 100 times the fraction with stronger > weaker.
    """

    indicators: list[float] = []
    for pair in level_pairs_db:
        if len(pair) != 2:
            return _invalid("each motion-loudness pair must contain two levels")
        stronger, weaker = map(float, pair)
        if not (_finite(stronger) and _finite(weaker)):
            return _invalid("non-finite event level")
        indicators.append(float(stronger > weaker))
    if not indicators:
        return _invalid("no valid localized event pair")
    return ScoreResult(True, 100.0 * sum(indicators) / len(indicators))


def log_attack_time(
    generated_attack_seconds: float,
    reference_attack_seconds: float,
    *,
    scale: float = 0.35,
    floor_seconds: float = 1e-4,
) -> ScoreResult:
    """Compute the I2AV Log Attack Time consistency score."""

    generated = float(generated_attack_seconds)
    reference = float(reference_attack_seconds)
    if not (_finite(generated) and _finite(reference)):
        return _invalid("non-finite attack duration")
    if generated <= 0.0 or reference <= 0.0:
        return _invalid("attack duration must be positive")
    generated = max(generated, floor_seconds)
    reference = max(reference, floor_seconds)
    error = abs(math.log(generated) - math.log(reference))
    return ScoreResult(True, 100.0 * math.exp(-error / scale))


def impact_decay(
    fit_r2: float,
    tail_residual_mae_db: float,
    peak_to_floor_db: float,
) -> ScoreResult:
    """Compute the continuous Impact Decay shape score."""

    r2 = float(fit_r2)
    residual = float(tail_residual_mae_db)
    dynamic_range = float(peak_to_floor_db)
    if not all(_finite(v) for v in (r2, residual, dynamic_range)):
        return _invalid("non-finite decay readout")
    if residual < 0.0 or dynamic_range <= 0.0:
        return _invalid("invalid decay residual or dynamic range")
    score = 100.0 * _clip(r2, 0.0, 1.0) * math.exp(
        -residual / max(dynamic_range, 1e-6)
    )
    return ScoreResult(True, score)


def rt60_consistency(audio_rt60: float, visual_rt60: float) -> ScoreResult:
    """Compute the 500 Hz audio-visual apparent-RT60 consistency score."""

    audio = float(audio_rt60)
    visual = float(visual_rt60)
    if not (_finite(audio) and _finite(visual)):
        return _invalid("non-finite RT60 proxy")
    if not 0.05 <= audio <= 5.0:
        return _invalid("audio RT60 is outside [0.05, 5.0] seconds")
    if not 0.08 <= visual <= 3.0:
        return _invalid("visual RT60 is outside [0.08, 3.0] seconds")
    ratio = audio / visual
    raw = (math.log(3.0) - abs(math.log(ratio))) / (
        math.log(3.0) - math.log(1.5)
    )
    return ScoreResult(True, 100.0 * _clip(raw, 0.0, 1.0))


def causality_violation(
    onset_delays_seconds: Iterable[float],
    *,
    early_threshold_seconds: float = -0.001,
) -> ScoreResult:
    """Score matched audio onset minus visual contact delays."""

    delays = [float(value) for value in onset_delays_seconds]
    if not delays:
        return _invalid("no reliable matched audio-visual event")
    if not all(_finite(value) for value in delays):
        return _invalid("non-finite onset delay")
    violations = sum(value < early_threshold_seconds for value in delays)
    return ScoreResult(True, 100.0 * (1.0 - violations / len(delays)))


def range_reference_db(
    relative_distances: Sequence[float], *, reference_index: int = 0
) -> list[float]:
    """Return the inverse-distance relative-level reference in decibels."""

    distances = [float(value) for value in relative_distances]
    if not distances or not 0 <= reference_index < len(distances):
        raise ValueError("a valid distance sequence and reference index are required")
    if not all(_finite(value) and value > 0.0 for value in distances):
        raise ValueError("relative distances must be positive and finite")
    reference = distances[reference_index]
    return [-20.0 * math.log10(value / reference) for value in distances]


def range_attenuation(window_r2_values: Iterable[float]) -> ScoreResult:
    """Map sign-aware local-window inverse-distance R2 readouts to 0--100.

    Tracking, relative-range estimation, sign resolution, exponent search, and
    window selection are performed by the receiver backend. The score API also
    accepts multiple independently selected readouts for batch-style callers.
    """

    values = [float(value) for value in window_r2_values]
    if not values:
        return _invalid("no valid range-scoring window")
    if not all(_finite(value) for value in values):
        return _invalid("non-finite range R2")
    clipped = [_clip(value, 0.0, 1.0) for value in values]
    return ScoreResult(True, 100.0 * sum(clipped) / len(clipped))


def native_receiver_score(value: float, *, evaluator: str) -> ScoreResult:
    """Validate a native Approach Gain or Lateral Stability readout.

    Their detector-dependent window metrics are emitted on [0, 1] by the
    receiver backend. This function performs the paper's final 0-100 mapping;
    it deliberately does not invent a replacement for the missing backend.
    """

    allowed = {"approach_gain", "lateral_stability"}
    if evaluator not in allowed:
        return _invalid(f"unsupported native receiver evaluator: {evaluator}")
    native = float(value)
    if not _finite(native):
        return _invalid("non-finite native receiver score")
    return ScoreResult(True, 100.0 * _clip(native, 0.0, 1.0))
