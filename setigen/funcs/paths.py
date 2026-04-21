"""
Sample signal paths for signal injection.

For any given starting frequency,
these functions map out the path of a signal as a function of time in
time-frequency space.
"""
from __future__ import annotations

from enum import Enum

import numpy as np
from astropy import units as u

from setigen import unit_utils
from setigen._typing import FrequencyPath, SeedLike


class SpreadType(str, Enum):
    """Supported spread distributions for simple RFI paths."""

    UNIFORM = "uniform"
    NORMAL = "normal"


class RfiType(str, Enum):
    """Supported RFI path families."""

    STATIONARY = "stationary"
    RANDOM_WALK = "random_walk"


def _coerce_spread_type(spread_type: str | SpreadType) -> SpreadType:
    """Normalize a user-supplied spread type.

    Args:
        spread_type: Raw spread-type selector.

    Returns:
        Normalized spread-type enum value.

    Raises:
        ValueError: If the spread type is unsupported.
    """
    if isinstance(spread_type, SpreadType):
        return spread_type
    try:
        return SpreadType(spread_type)
    except ValueError as exc:
        raise ValueError(f"'{spread_type}' is not a valid spread type!") from exc


def _coerce_rfi_type(rfi_type: str | RfiType) -> RfiType:
    """Normalize a user-supplied RFI type.

    Args:
        rfi_type: Raw RFI-type selector.

    Returns:
        Normalized RFI-type enum value.
    """
    if isinstance(rfi_type, RfiType):
        return rfi_type
    if rfi_type == RfiType.RANDOM_WALK.value:
        return RfiType.RANDOM_WALK
    return RfiType.STATIONARY


def constant_path(f_start: float | u.Quantity, drift_rate: float | u.Quantity) -> FrequencyPath:
    """Return a constant-drift path.

    Args:
        f_start: Starting center frequency.
        drift_rate: Doppler drift rate.

    Returns:
        Path callable.
    """
    f_start = unit_utils.get_value(f_start, u.Hz)
    drift_rate = unit_utils.get_value(drift_rate, u.Hz / u.s)

    def path(t):
        return f_start + drift_rate * t
    return path


def squared_path(f_start: float | u.Quantity, drift_rate: float | u.Quantity) -> FrequencyPath:
    """Return a quadratic drift path.

    Args:
        f_start: Starting center frequency.
        drift_rate: Initial drift-rate slope.

    Returns:
        Path callable.
    """
    f_start = unit_utils.get_value(f_start, u.Hz)
    drift_rate = unit_utils.get_value(drift_rate, u.Hz / u.s)

    def path(t):
        return f_start + 0.5 * drift_rate * t**2
    return path


def sine_path(
    f_start: float | u.Quantity,
    drift_rate: float | u.Quantity,
    period: float | u.Quantity,
    amplitude: float | u.Quantity,
) -> FrequencyPath:
    """Return a sinusoidally modulated drift path.

    Args:
        f_start: Starting center frequency.
        drift_rate: Doppler drift rate.
        period: Modulation period.
        amplitude: Modulation amplitude.

    Returns:
        Path callable.
    """
    f_start = unit_utils.get_value(f_start, u.Hz)
    drift_rate = unit_utils.get_value(drift_rate, u.Hz / u.s)
    period = unit_utils.get_value(period, u.s)
    amplitude = unit_utils.get_value(amplitude, u.Hz)

    def path(t):
        return f_start + amplitude * np.sin(2*np.pi*t/period) + drift_rate * t
    return path


def simple_rfi_path(
    f_start: float | u.Quantity,
    drift_rate: float | u.Quantity,
    spread: float | u.Quantity,
    spread_type: str | SpreadType = 'uniform', 
    rfi_type: str | RfiType = 'stationary',
    seed: SeedLike = None,
) -> FrequencyPath:
    """Return a simple randomized RFI path.

    Args:
        f_start: Starting center frequency.
        drift_rate: Doppler drift rate.
        spread: Width of frequency variations.
        spread_type: Distribution used for per-time-step offsets.
        rfi_type: Whether offsets are stationary or cumulative.
        seed: Random seed or generator.

    Returns:
        Path callable.
    """
    rng = np.random.default_rng(seed)
    f_start = unit_utils.get_value(f_start, u.Hz)
    drift_rate = unit_utils.get_value(drift_rate, u.Hz / u.s)
    spread = unit_utils.get_value(spread, u.Hz)
    resolved_spread_type = _coerce_spread_type(spread_type)
    resolved_rfi_type = _coerce_rfi_type(rfi_type)

    def path(t):
        if resolved_spread_type is SpreadType.UNIFORM:
            f_offset = rng.uniform(-spread / 2., spread / 2., size=t.shape)
        else:
            factor = 2 * np.sqrt(2 * np.log(2))
            f_offset = rng.normal(0, spread / factor, size=t.shape)
            
        if resolved_rfi_type is RfiType.RANDOM_WALK:
            f_offset = np.cumsum(f_offset)
        return f_start + drift_rate * t + f_offset
    return path
