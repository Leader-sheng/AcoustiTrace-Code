"""Paper-aligned range-guidance objective independent of a diffusion runtime."""

from __future__ import annotations

import math
from typing import Sequence


def deterministic_pairs(length: int, max_pairs: int = 4096) -> list[tuple[int, int]]:
    if length < 2:
        return []
    pairs = [(i, j) for i in range(length) for j in range(i + 1, length)]
    if len(pairs) <= max_pairs:
        return pairs
    if max_pairs < 1:
        return []
    if max_pairs == 1:
        return [pairs[0]]
    last = len(pairs) - 1
    indices = [int(step * last / (max_pairs - 1)) for step in range(max_pairs)]
    return [pairs[index] for index in indices]


def smooth_l1(value: float, beta: float) -> float:
    absolute = abs(value)
    if absolute < beta:
        return 0.5 * absolute * absolute / beta
    return absolute - 0.5 * beta


def decoded_mel_pairwise_loss(
    envelope: Sequence[float],
    distance: Sequence[float],
    *,
    gamma: float = 1.0,
    epsilon: float = 1e-6,
    audio_rms_gate: float = 1e-5,
    huber_beta: float = 0.05,
    max_pairs: int = 4096,
) -> dict[str, float | int]:
    """Evaluate the robust log-envelope/log-distance guidance objective.

    The caller is responsible for decoding the audio x0 latent and interpolating
    the visual distance trajectory to the decoded-mel time axis.
    """

    energy = [float(value) for value in envelope]
    ranges = [float(value) for value in distance]
    if len(energy) != len(ranges):
        raise ValueError("envelope and distance must have equal length")
    if not all(math.isfinite(value) and value >= 0.0 for value in energy):
        raise ValueError("envelope values must be finite and nonnegative")
    if not all(math.isfinite(value) and value > 0.0 for value in ranges):
        raise ValueError("distance values must be finite and positive")
    residuals: list[float] = []
    for i, j in deterministic_pairs(len(energy), max_pairs=max_pairs):
        if energy[i] <= audio_rms_gate or energy[j] <= audio_rms_gate:
            continue
        residual = (
            math.log(energy[i] + epsilon)
            - math.log(energy[j] + epsilon)
            - gamma * (math.log(ranges[j] + epsilon) - math.log(ranges[i] + epsilon))
        )
        residuals.append(residual)
    if not residuals:
        raise ValueError("no pair passes the decoded-mel RMS gate")
    losses = [smooth_l1(value, huber_beta) for value in residuals]
    return {
        "loss": sum(losses) / len(losses),
        "n_pairs": len(losses),
        "mean_abs_residual": sum(map(abs, residuals)) / len(residuals),
    }

