"""
Sample spectral profiles for signal injection.

For any given time sample,
these functions map out the intensity in the frequency direction (centered at
a particular frequency).
"""
import numpy as np
from astropy import units as u

from setigen import unit_utils
from setigen.funcs import func_utils


def box_f_profile(width):
    """
    Square intensity profile in the frequency direction.
    """
    width = unit_utils.get_value(width, u.Hz)

    def f_profile(f, f_center):
        return (np.abs(f-f_center) < width / 2).astype(int)
    return f_profile


def gaussian_f_profile(width):
    """
    Gaussian profile; width is the FWHM of the profile.
    """
    width = unit_utils.get_value(width, u.Hz)
    factor = 2 * np.sqrt(2 * np.log(2))
    sigma = width / factor

    def f_profile(f, f_center):
        return func_utils.gaussian(f, f_center, sigma)
    return f_profile


def multiple_gaussian_f_profile(width):
    """
    Example adding multiple Gaussians in the frequency direction.
    """
    width = unit_utils.get_value(width, u.Hz)
    factor = 2 * np.sqrt(2 * np.log(2))
    sigma = width / factor

    def f_profile(f, f_center):
        # Offsets by 100 Hz @ a quarter intensity, absolutely arbitrarily
        return func_utils.gaussian(f, f_center - 100, sigma) / 4 \
            + func_utils.gaussian(f, f_center, sigma) \
            + func_utils.gaussian(f, f_center + 100, sigma) / 4
    return f_profile
