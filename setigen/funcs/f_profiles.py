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
    
    Parameters
    ----------
    width : float or astropy.Quantity
        Width of signal

    Return
    ------
    f_profile : func
    """
    width = unit_utils.get_value(width, u.Hz)

    def f_profile(f, f_center):
        return (np.abs(f - f_center) < width / 2).astype(int)
    return f_profile


def gaussian_f_profile(width):
    """
    Gaussian profile.
    
    Parameters
    ----------
    width : float or astropy.Quantity
        FWHM of Gaussian profile

    Return
    ------
    f_profile : func
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
    
    Parameters
    ----------
    width : float or astropy.Quantity
        FWHM of Gaussian profile

    Return
    ------
    f_profile : func
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


def lorentzian_f_profile(width):
    """
    Lorentzian profile.
    
    Parameters
    ----------
    width : float or astropy.Quantity
        FWHM of Lorentzian profile

    Return
    ------
    f_profile : func
    """
    width = unit_utils.get_value(width, u.Hz)
    gamma = width / 2

    def f_profile(f, f_center):
        return func_utils.lorentzian(f, f_center, gamma)
    return f_profile


def voigt_f_profile(g_width, l_width):
    """
    Voigt profile.
    
    Further information here: https://en.wikipedia.org/wiki/Voigt_profile.
    
    Parameters
    ----------
    g_width : float or astropy.Quantity
        FWHM of Gaussian profile
    l_width : float or astropy.Quantity
        FWHM of Lorentzian profile

    Return
    ------
    f_profile : func
    """
    g_width = unit_utils.get_value(g_width, u.Hz)
    factor = 2 * np.sqrt(2 * np.log(2))
    sigma = g_width / factor
    
    l_width = unit_utils.get_value(l_width, u.Hz)
    gamma = l_width / 2

    def f_profile(f, f_center):
        return func_utils.voigt(f, f_center, sigma, gamma) / func_utils.voigt(f_center, f_center, sigma, gamma)
    return f_profile


def sinc2_f_profile(width, width_mode='fwhm', trunc=True):
    """
    Sinc squared profile; width is the FWHM of the squared normalized sinc function.
    
    The trunc parameter controls whether or not the sinc squared profile is 
    truncated at the first root (e.g. zeroed out for more distant frequencies).

    Note that you can model an ideal cosine signal by using 
    :code:`width=2*frame.df`, :code:`width_mode="crossing"`, and 
    :code:`trunc=False`.
    
    Parameters
    ----------
    width : float or astropy.Quantity
        Signal width, in Hz
    width_mode : {"fwhm", "crossing"}, default: "fwhm"
        How to interpret :code:`width`. Can be "fwhm" or "crossing", for 
        zero crossing.
    trunc : bool, default: True
        Whether to truncate signal after first zero crossing

    Return
    ------
    f_profile : func
    """
    width = unit_utils.get_value(width, u.Hz)
    
    # Using the numerical solution for the FWHM
    if width_mode == 'fwhm':
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
    return f_profile