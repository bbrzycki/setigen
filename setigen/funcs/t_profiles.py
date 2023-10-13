"""
Sample intensity profiles for signal injection.

These functions calculate the signal intensity and variation in the time
direction.
"""
import sys
import numpy as np
from astropy import units as u

from setigen import unit_utils
from setigen.funcs import func_utils


def constant_t_profile(level=1):
    """
    Constant intensity profile.
    
    Parameters
    ----------
    level : float, default: 1
        Intensity level

    Return
    ------
    t_profile : func
    """
    def t_profile(t):
        if isinstance(t, (np.ndarray, list)):
            shape = np.array(t).shape
        else:
            return level
        return np.full(shape, level)
    return t_profile


def sine_t_profile(period, phase=0, amplitude=1, level=1):
    """
    Intensity varying as a sine curve.
    
    Parameters
    ----------
    period : float or astropy.Quantity
        Modulation period
    phase : float, default: 0
        Modulation phase
    amplitude : float, default: 1
        Modulation amplitude
    level : float, default: 1
        Mean intensity level

    Return
    ------
    t_profile : func
    """
    period = unit_utils.get_value(period, u.s)

    def t_profile(t):
        return amplitude * np.sin(2 * np.pi * (t + phase) / period) + level
    return t_profile


def periodic_gaussian_t_profile(pulse_width,
                                period,
                                phase=0,
                                pulse_offset_width=0,
                                pulse_direction='rand',
                                pnum=3,
                                amplitude=1,
                                level=1,
                                min_level=0):
    """
    Intensity varying as Gaussian pulses, allowing for variation in the arrival
    time of each pulse.
    
    Parameters
    ----------
    pulse_width : float or astropy.Quantity
        FWHM width of individual pulses
    period : float or astropy.Quantity
        Baseline modulation period
    phase : float or astropy.Quantity, default: 0
        Baseline modulation phase
    pulse_offset_width : float or astropy.Quantity, default: 0
        FWHM of timing variation from the modulation period
    pulse_direction : {"rand", "up", "down"}, default: "rand"
        Whether the intensity increases or decreases from the baseline level
    pnum : float or astropy.Quantity, default: 3
        Number of Gaussians pulses to consider when calculating the intensity 
        at each timestep. The higher this number, the more accurate the
        intensities.
    amplitude : float, default: 1
        Pulse magnitude
    level : float, default: 1
        Baseline intensity level
    min_level : float, default: 0
        Minimum intensity level

    Return
    ------
    t_profile : func
    """
    period = unit_utils.get_value(period, u.s)

    factor = 2 * np.sqrt(2 * np.log(2))
    pulse_offset_sigma = unit_utils.get_value(pulse_offset_width, u.s) / factor
    pulse_sigma = unit_utils.get_value(pulse_width, u.s) / factor

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
                               np.random.normal(0,
                                                pulse_offset_sigma,
                                                unique_center_ks.shape)))
        get_offsets = np.vectorize(lambda x: offset_dict[x])

        # Calculate the signs for each pulse
        sign_list = []
        for c in unique_center_ks:
            x = np.random.uniform(0, 1)
            if (pulse_direction == 'up'
                    or pulse_direction == 'rand' and x < 0.5):
                sign_list.append(1)
            elif pulse_direction == 'down' or pulse_direction == 'rand':
                sign_list.append(-1)
            else:
                sys.exit('Invalid pulse direction!')
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
