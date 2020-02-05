import numpy as np
from scipy.special import wofz


def gaussian(x, x0, sigma):
    return np.exp(-np.power(x - x0, 2.) / (2 * np.power(sigma, 2.)))


def lorentzian(x, x0, gamma):
    return 1 / (1 + np.power((x - x0) / gamma, 2))


def voigt(x, x0, sigma, gamma):
    if sigma == 0:
        return lorentzian(x, x0, gamma)
    if gamma == 0:
        return gaussian(x, x0, sigma)
    return np.real(wofz(((x - x0) + 1j * gamma) / sigma / np.sqrt(2)))


def voigt_fwhm(g_width, l_width):
    """
    Accurate to 0.0003% for a pure Lorentzian profile, precise for a pure Gaussian.

    Source: https://en.wikipedia.org/wiki/Voigt_profile.
    """
    return 0.5346 * l_width + np.sqrt(0.2166 * l_width**2 + g_width**2)
