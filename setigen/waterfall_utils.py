import sys
from pathlib import PurePath
import numpy as np
from blimpy import Waterfall


def max_freq(waterfall):
    """
    Return central frequency of the highest-frequency bin in a .fil file.

    Parameters
    ----------
    waterfall : str or Waterfall
        Name of filterbank file or Waterfall object

    Returns
    -------
    fmax : float
        Maximum frequency in data
    """
    return np.sort(get_fs(waterfall))[-1]


def min_freq(waterfall):
    """
    Return central frequency of the lowest-frequency bin in a .fil file.

    Parameters
    ----------
    waterfall : str or Waterfall
        Name of filterbank file or Waterfall object

    Returns
    -------
    fmin : float
        Minimum frequency in data
    """
    return np.sort(get_fs(waterfall))[0]


def get_data(waterfall, db=False):
    """
    Get time-frequency data from filterbank file as a 2d NumPy array.

    Note: when multiple Stokes parameters are supported, this will have to
    be expanded.

    Parameters
    ----------
    waterfall : str or Waterfall
        Name of filterbank file or Waterfall object

    Returns
    -------
    data : ndarray
        Time-frequency data
    """
    if isinstance(waterfall, (str, PurePath)):
        waterfall = Waterfall(waterfall)
    elif not isinstance(waterfall, Waterfall):
        raise ValueError('Invalid data file!')

    if db:
        return 10 * np.log10(waterfall.data[:, 0, :])

    return waterfall.data[:, 0, :]


def get_fs(waterfall):
    """
    Get frequency values from filterbank file.

    Parameters
    ----------
    waterfall : str or Waterfall
        Name of filterbank file or Waterfall object

    Returns
    -------
    fs : ndarray
        Frequency values
    """
    if isinstance(waterfall, (str, PurePath)):
        waterfall = Waterfall(waterfall, load_data=False)
    elif not isinstance(waterfall, Waterfall):
        raise ValueError('Invalid data file!')

    fch1 = waterfall.header['fch1']
    df = waterfall.header['foff']
    fchans = waterfall.header['nchans']

    return np.arange(fch1, fch1 + fchans * df, df)


def get_ts(waterfall):
    """
    Get time values from filterbank file.

    Parameters
    ----------
    waterfall : str or Waterfall
        Name of filterbank file or Waterfall object

    Returns
    -------
    ts : ndarray
        Time values
    """
    if isinstance(waterfall, (str, PurePath)):
        waterfall = Waterfall(waterfall, load_data=False)
    elif not isinstance(waterfall, Waterfall):
        raise ValueError('Invalid data file!')

    tsamp = waterfall.header['tsamp']
    tchans = waterfall.container.selection_shape[0]

    return np.arange(0, tchans * tsamp, tsamp)
