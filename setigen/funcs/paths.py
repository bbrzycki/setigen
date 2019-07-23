import sys
import numpy as np
from astropy import units as u

from setigen import unit_utils


def constant_path(f_start, drift_rate):
    f_start = unit_utils.get_value(f_start, u.Hz)
    drift_rate = unit_utils.get_value(drift_rate, u.Hz / u.s)

    def path(t):
        return f_start + drift_rate * t
    return path


def squared_path(f_start, drift_rate):
    f_start = unit_utils.get_value(f_start, u.Hz)
    drift_rate = unit_utils.get_value(drift_rate, u.Hz / u.s)

    def path(t):
        return f_start + drift_rate * t**2
    return path


def sine_path(f_start, drift_rate, period, amplitude):
    f_start = unit_utils.get_value(f_start, u.Hz)
    drift_rate = unit_utils.get_value(drift_rate, u.Hz / u.s)
    period = unit_utils.get_value(period, u.s)
    amplitude = unit_utils.get_value(amplitude, u.Hz)

    def path(t):
        return f_start + amplitude * np.sin(2*np.pi*t/period) + drift_rate * t
    return path


def choppy_rfi_path(f_start, drift_rate, spread, spread_type='uniform'):
    f_start = unit_utils.get_value(f_start, u.Hz)
    drift_rate = unit_utils.get_value(drift_rate, u.Hz / u.s)
    spread = unit_utils.get_value(spread, u.Hz)

    def path(t):
        if spread_type == 'uniform':
            f_offset = np.random.uniform(-spread / 2., spread / 2., t.shape)
        elif spread_type == 'gaussian':
            f_offset = np.random.normal(0, spread, t.shape)
        else:
            sys.exit('%s is not a valid spread type!' % spread_type)
        return f_start + drift_rate * t + f_offset
    return path
