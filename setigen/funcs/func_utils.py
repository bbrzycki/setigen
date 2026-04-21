from __future__ import annotations

import numpy as np
from scipy.special import wofz


def gaussian(x: np.ndarray | float, x0: np.ndarray | float, sigma: float) -> np.ndarray | float:
    """Evaluate a Gaussian profile.

    Args:
        x: Coordinate values.
        x0: Gaussian center.
        sigma: Gaussian standard deviation.

    Returns:
        Gaussian profile values at `x`.
    """
    return np.exp(-np.power(x - x0, 2.) / (2 * np.power(sigma, 2.)))


def lorentzian(x: np.ndarray | float, x0: np.ndarray | float, gamma: float) -> np.ndarray | float:
    """Evaluate a Lorentzian profile.

    Args:
        x: Coordinate values.
        x0: Lorentzian center.
        gamma: Lorentzian half-width parameter.

    Returns:
        Lorentzian profile values at `x`.
    """
    return 1 / (1 + np.power((x - x0) / gamma, 2))


def voigt(
    x: np.ndarray | float,
    x0: np.ndarray | float,
    sigma: float,
    gamma: float,
) -> np.ndarray | float:
    """Evaluate a Voigt profile.

    Args:
        x: Coordinate values.
        x0: Profile center.
        sigma: Gaussian standard deviation.
        gamma: Lorentzian half-width parameter.

    Returns:
        Voigt profile values at `x`.
    """
    if sigma == 0:
        return lorentzian(x, x0, gamma)
    if gamma == 0:
        return gaussian(x, x0, sigma)
    return np.real(wofz(((x - x0) + 1j * gamma) / sigma / np.sqrt(2)))


def voigt_fwhm(g_width: float, l_width: float) -> float:
    """Approximate the FWHM of a Voigt profile.

    Args:
        g_width: Gaussian FWHM.
        l_width: Lorentzian FWHM.

    Returns:
        Approximate Voigt FWHM.
    """
    return 0.5346 * l_width + np.sqrt(0.2166 * l_width**2 + g_width**2)
