"""
Sample signal paths for signal injection.

For any given starting frequency,
these functions map out the path of a signal as a function of time in
time-frequency space.
"""
import sys
import numpy as np
from astropy import units as u

from setigen import unit_utils


def constant_path(f_start, drift_rate):
    """
    Constant drift rate.
    
    Parameters
    ----------
    f_start : float or astropy.Quantity
        Starting center frequency
    drift_rate : float or astropy.Quantity
        Doppler drift rate

    Return
    ------
    path : func
    """
    f_start = unit_utils.get_value(f_start, u.Hz)
    drift_rate = unit_utils.get_value(drift_rate, u.Hz / u.s)

    def path(t):
        return f_start + drift_rate * t
    return path


def squared_path(f_start, drift_rate):
    """
    Quadratic signal path; drift_rate here only refers to the starting slope.
    
    Parameters
    ----------
    f_start : float or astropy.Quantity
        Starting center frequency
    drift_rate : float or astropy.Quantity
        Doppler drift rate

    Return
    ------
    path : func
    """
    f_start = unit_utils.get_value(f_start, u.Hz)
    drift_rate = unit_utils.get_value(drift_rate, u.Hz / u.s)

    def path(t):
        return f_start + 0.5 * drift_rate * t**2
    return path


def sine_path(f_start, drift_rate, period, amplitude):
    """
    Sine path in time-frequency space.
    
    Parameters
    ----------
    f_start : float or astropy.Quantity
        Starting center frequency
    drift_rate : float or astropy.Quantity
        Doppler drift rate
    period : float or astropy.Quantity
        Modulation period
    amplitude : float or astropy.Quantity
        Modulation amplitude

    Return
    ------
    path : func
    """
    f_start = unit_utils.get_value(f_start, u.Hz)
    drift_rate = unit_utils.get_value(drift_rate, u.Hz / u.s)
    period = unit_utils.get_value(period, u.s)
    amplitude = unit_utils.get_value(amplitude, u.Hz)

    def path(t):
        return f_start + amplitude * np.sin(2*np.pi*t/period) + drift_rate * t
    return path


def simple_rfi_path(f_start, drift_rate, spread, spread_type='uniform', 
                    rfi_type='stationary'):
    """
    A crude simulation of one style of RFI that shows up, in which the signal
    jumps around in frequency. This method samples the center frequency for
    each time sample from either a uniform or normal distribution. 
    
    Parameters
    ----------
    f_start : float or astropy.Quantity
        Starting center frequency
    drift_rate : float or astropy.Quantity
        Doppler drift rate
    spread : float or astropy.Quantity
        Range of center frequency variations
    spread_type : {"uniform", "normal"}, default: "uniform"
        Type of frequency variation
    rfi_type : {"stationary", "random_walk"}, default: "stationary"
        The "stationary" option only offsets with respect to a straight-line 
        path, but "random_walk" accumulates frequency offsets over time.

    Return
    ------
    path : func
    """
    f_start = unit_utils.get_value(f_start, u.Hz)
    drift_rate = unit_utils.get_value(drift_rate, u.Hz / u.s)
    spread = unit_utils.get_value(spread, u.Hz)

    def path(t):
        if spread_type == 'uniform':
            f_offset = np.random.uniform(-spread / 2., spread / 2., size=t.shape)
        elif spread_type == 'normal':
            factor = 2 * np.sqrt(2 * np.log(2))
            f_offset = np.random.normal(0, spread / factor, size=t.shape)
        else:
            sys.exit('{} is not a valid spread type!'.format(spread_type))
            
        if rfi_type == 'random_walk':
            f_offset = np.cumsum(f_offset)
        return f_start + drift_rate * t + f_offset
    return path