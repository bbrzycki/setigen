"""
Sample spectral profiles for signal injection.

For any given time sample,
these functions map out the intensity in the frequency direction (centered at
a particular frequency).
"""
from __future__ import annotations

from enum import Enum

import numpy as np
from astropy import units as u

from setigen import unit_utils
from setigen._typing import FrequencyProfile
from setigen.funcs import func_utils


def _with_profile_metadata(profile: FrequencyProfile, **metadata: float | bool | str) -> FrequencyProfile:
    """Attach setigen-specific support metadata to a frequency profile.

    Args:
        profile: Frequency-profile callable to annotate.
        **metadata: Metadata describing finite support or cutoff parameters.

    Returns:
        The same frequency-profile callable.
    """
    profile._setigen_profile_metadata = metadata
    return profile


class WidthMode(str, Enum):
    """Supported width interpretations for sinc-squared profiles."""

    CROSSING = "crossing"
    FWHM = "fwhm"


def _coerce_width_mode(width_mode: str | WidthMode) -> WidthMode:
    """Normalize a user-supplied width mode.

    Args:
        width_mode: Raw width-mode selector.

    Returns:
        Normalized width-mode enum value.
    """
    if isinstance(width_mode, WidthMode):
        return width_mode
    if width_mode == WidthMode.FWHM.value:
        return WidthMode.FWHM
    return WidthMode.CROSSING


def box_f_profile(width: float | u.Quantity) -> FrequencyProfile:
    """Return a box spectral profile.

    Args:
        width: Signal width.

    Returns:
        Frequency-profile callable.
    """
    width = unit_utils.get_value(width, u.Hz)

    def f_profile(f, f_center):
        return (np.abs(f - f_center) < width / 2).astype(int)
    return _with_profile_metadata(f_profile,
                                  finite_support=True,
                                  half_width=width / 2)


def gaussian_f_profile(width: float | u.Quantity) -> FrequencyProfile:
    """Return a Gaussian spectral profile.

    Args:
        width: Gaussian FWHM.

    Returns:
        Frequency-profile callable.
    """
    width = unit_utils.get_value(width, u.Hz)
    factor = 2 * np.sqrt(2 * np.log(2))
    sigma = width / factor

    def f_profile(f, f_center):
        return func_utils.gaussian(f, f_center, sigma)
    return _with_profile_metadata(f_profile,
                                  finite_support=False,
                                  profile_type="gaussian",
                                  sigma=sigma)


def multiple_gaussian_f_profile(width: float | u.Quantity) -> FrequencyProfile:
    """Return a multi-component Gaussian spectral profile.

    Args:
        width: Gaussian FWHM.

    Returns:
        Frequency-profile callable.
    """
    width = unit_utils.get_value(width, u.Hz)
    factor = 2 * np.sqrt(2 * np.log(2))
    sigma = width / factor

    def f_profile(f, f_center):
        # Offsets by 100 Hz @ a quarter intensity, absolutely arbitrarily
        return func_utils.gaussian(f, f_center - 100, sigma) / 4 \
            + func_utils.gaussian(f, f_center, sigma) \
            + func_utils.gaussian(f, f_center + 100, sigma) / 4
    return _with_profile_metadata(f_profile,
                                  finite_support=False,
                                  profile_type="multiple_gaussian",
                                  sigma=sigma,
                                  max_offset=100)


def lorentzian_f_profile(width: float | u.Quantity) -> FrequencyProfile:
    """Return a Lorentzian spectral profile.

    Args:
        width: Lorentzian FWHM.

    Returns:
        Frequency-profile callable.
    """
    width = unit_utils.get_value(width, u.Hz)
    gamma = width / 2

    def f_profile(f, f_center):
        return func_utils.lorentzian(f, f_center, gamma)
    return _with_profile_metadata(f_profile,
                                  finite_support=False,
                                  profile_type="lorentzian",
                                  gamma=gamma)


def voigt_f_profile(g_width: float | u.Quantity, l_width: float | u.Quantity) -> FrequencyProfile:
    """Return a Voigt spectral profile.

    Args:
        g_width: Gaussian FWHM.
        l_width: Lorentzian FWHM.

    Returns:
        Frequency-profile callable.
    """
    g_width = unit_utils.get_value(g_width, u.Hz)
    factor = 2 * np.sqrt(2 * np.log(2))
    sigma = g_width / factor
    
    l_width = unit_utils.get_value(l_width, u.Hz)
    gamma = l_width / 2

    def f_profile(f, f_center):
        return func_utils.voigt(f, f_center, sigma, gamma) / func_utils.voigt(f_center, f_center, sigma, gamma)
    return _with_profile_metadata(f_profile,
                                  finite_support=False,
                                  profile_type="voigt",
                                  sigma=sigma,
                                  gamma=gamma)


def sinc2_f_profile(
    width: float | u.Quantity,
    width_mode: str | WidthMode = "crossing",
    trunc: bool = True,
) -> FrequencyProfile:
    """Return a sinc-squared spectral profile.

    Args:
        width: Signal width in Hz.
        width_mode: Whether `width` is interpreted as zero-crossing or FWHM.
        trunc: Whether to truncate after the first zero crossing.

    Returns:
        Frequency-profile callable.
    """
    width = unit_utils.get_value(width, u.Hz)
    resolved_width_mode = _coerce_width_mode(width_mode)
    
    # Using the numerical solution for the FWHM
    if resolved_width_mode is WidthMode.FWHM:
        zero_crossing = (width / 2) / 0.442946470689452
    else:
        zero_crossing = width / 2
    
    def f_profile(f, f_center):
        if trunc:
            return np.where(np.abs(f - f_center) < zero_crossing, 
                            np.sinc((f - f_center) / zero_crossing),
                            0)**2
        else:
            return np.sinc((f - f_center) / zero_crossing)**2
    return _with_profile_metadata(f_profile,
                                  finite_support=trunc,
                                  profile_type="sinc2",
                                  half_width=zero_crossing)
