"""
Sample intensity profiles for signal injection.

These functions calculate the signal intensity and variation in the time
direction.
"""
from __future__ import annotations

from enum import Enum

import numpy as np
from astropy import units as u

from setigen import unit_utils
from setigen._typing import SeedLike, TimeProfile
from setigen.funcs import func_utils


class PulseDirection(str, Enum):
    """Supported pulse directions for periodic Gaussian profiles."""

    RANDOM = "rand"
    UP = "up"
    DOWN = "down"


def _coerce_pulse_direction(pulse_direction: str | PulseDirection) -> PulseDirection:
    """Normalize a user-supplied pulse direction.

    Args:
        pulse_direction: Raw pulse-direction selector.

    Returns:
        Normalized pulse-direction enum value.

    Raises:
        ValueError: If the pulse direction is unsupported.
    """
    if isinstance(pulse_direction, PulseDirection):
        return pulse_direction
    try:
        return PulseDirection(pulse_direction)
    except ValueError as exc:
        raise ValueError(f"Invalid pulse direction: {pulse_direction!r}") from exc


def constant_t_profile(level: float = 1) -> TimeProfile:
    """Return a constant intensity profile.

    Args:
        level: Constant intensity level.

    Returns:
        Time-profile callable.
    """
    def t_profile(t):
        if isinstance(t, (np.ndarray, list)):
            shape = np.array(t).shape
        else:
            return level
        return np.full(shape, level)
    return t_profile


def sine_t_profile(
    period: float | u.Quantity,
    phase: float = 0,
    amplitude: float = 1,
    level: float = 1,
) -> TimeProfile:
    """Return a sinusoidal intensity profile.

    Args:
        period: Modulation period.
        phase: Modulation phase.
        amplitude: Modulation amplitude.
        level: Mean intensity level.

    Returns:
        Time-profile callable.
    """
    period = unit_utils.get_value(period, u.s)

    def t_profile(t):
        return amplitude * np.sin(2 * np.pi * (t + phase) / period) + level
    return t_profile


def periodic_gaussian_t_profile(pulse_width: float | u.Quantity,
                                period: float | u.Quantity,
                                phase: float | u.Quantity = 0,
                                pulse_offset_width: float | u.Quantity = 0,
                                pulse_direction: str | PulseDirection = 'rand',
                                pnum: int = 3,
                                amplitude: float = 1,
                                level: float = 1,
                                min_level: float = 0,
                                seed: SeedLike = None) -> TimeProfile:
    """Return a periodic Gaussian-pulse intensity profile.

    Args:
        pulse_width: FWHM of individual pulses.
        period: Baseline modulation period.
        phase: Baseline modulation phase.
        pulse_offset_width: FWHM of timing jitter.
        pulse_direction: Whether pulses go up, down, or randomly both.
        pnum: Number of neighboring pulses to include in the calculation.
        amplitude: Pulse magnitude.
        level: Baseline intensity level.
        min_level: Minimum allowed intensity level.
        seed: Random seed or generator.

    Returns:
        Time-profile callable.
    """
    rng = np.random.default_rng(seed)
    period = unit_utils.get_value(period, u.s)

    factor = 2 * np.sqrt(2 * np.log(2))
    pulse_offset_sigma = unit_utils.get_value(pulse_offset_width, u.s) / factor
    pulse_sigma = unit_utils.get_value(pulse_width, u.s) / factor
    resolved_pulse_direction = _coerce_pulse_direction(pulse_direction)

    def t_profile(t):
        # This gives an array of length len(t)
        center_ks = np.round((t + phase) / period - 1 / 4.)

        # This conditional could be written in one line, but that obfuscates
        # the code. Here we determine which pulse centers need to be considered
        # for each time sample (e.g. the closest pnum pulses)
        temp = pnum // 2
        if pnum % 2 == 1:
            center_ks = np.array([center_ks + 1 * i
                                  for i in np.arange(-temp, temp + 1)])
        else:
            center_ks = np.array([center_ks + 1 * i
                                  for i in np.arange(-temp + 1, temp + 1)])
        # Here center_ks.shape = (pnum, len(t)), of ints
        centers = (4. * center_ks + 1.) / 4. * period - phase

        # Calculate unique offsets per pulse and add to centers of Gaussians
        # Each element in unique_center_ks corresponds to a distinct (tracked)
        # pulse
        unique_center_ks = np.unique(center_ks)

        # Apply the pulse offset to each tracked pulse
        offset_dict = dict(zip(unique_center_ks,
                               rng.normal(0,
                                          pulse_offset_sigma,
                                          unique_center_ks.shape)))
        get_offsets = np.vectorize(lambda x: offset_dict[x])

        # Calculate the signs for each pulse
        sign_list = []
        for c in unique_center_ks:
            x = rng.uniform(0, 1)
            if (resolved_pulse_direction is PulseDirection.UP
                    or resolved_pulse_direction is PulseDirection.RANDOM and x < 0.5):
                sign_list.append(1)
            elif (resolved_pulse_direction is PulseDirection.DOWN
                  or resolved_pulse_direction is PulseDirection.RANDOM):
                sign_list.append(-1)
        sign_dict = dict(zip(unique_center_ks, sign_list))
        get_signs = np.vectorize(lambda x: sign_dict[x])

        # Apply the previously computed variations and total to compute
        # intensities
        centers += get_offsets(center_ks)
        center_signs = zip(centers, get_signs(center_ks))

        intensity = 0
        for c, sign in center_signs:
            intensity += sign * amplitude * func_utils.gaussian(t,
                                                                c,
                                                                pulse_sigma)
        intensity += level
        return np.maximum(min_level, intensity)
    return t_profile
