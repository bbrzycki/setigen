import numpy as np
import sys

def generate(ts,
             fs,
             path,
             t_profile,
             f_profile,
             bp_profile):
    """
    Generates synthetic signal based on given path in time-frequency domain and
    brightness profiles in time and frequency directions.

    Args:
        ts, NumPy array of time samples
        fs, NumPy array of frequency samples
        path, function in time that returns frequencies
        t_profile, function in time that returns an intensity (scalar)
        f_profile, function in frequency that returns an intensity (scalar),
            relative to the signal frequency within a time sample
        bp_profile, function in frequency that returns an intensity (scalar)

    Return:
        signal, two-dimensional NumPy array that only contains synthetic signal
            data
    """
    ff, tt = np.meshgrid(fs, ts)
    return t_profile(tt) * f_profile(ff, path(tt)) * bp_profile(ff)
