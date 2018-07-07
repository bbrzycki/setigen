import sys
import os
import errno
import numpy as np
from blimpy import read_header, Waterfall, Filterbank

def maxfreq(input_fn):
    """Return central frequency of the highest-frequency bin in a .fil file.

    """
    return read_header(input_fn)[b'fch1']

def minfreq(input_fn):
    """Return central frequency of the lowest-frequency bin in a .fil file.

    """
    fch1 = read_header(input_fn)[b'fch1']
    nchans = read_header(input_fn)[b'nchans']
    ch_bandwidth = read_header(input_fn)[b'foff']
    return fch1 + nchans * ch_bandwidth

def get_data(input_fn):
    """Gets time-frequency data from filterbank file as a 2d NumPy array.

    Parameters
    ----------
    input_fn : str
        Name of filterbank file

    Returns
    -------
    data : ndarray
        Time-frequency data
    """
    fil = Waterfall(input_fn)
    return np.squeeze(fil.data)

def get_fs(input_fn):
    """Gets frequency values from filterbank file.

    Parameters
    ----------
    input_fn : str
        Name of filterbank file

    Returns
    -------
    fs : ndarray
        Frequency values
    """
    fch1 = read_header(input_fn)[b'fch1']
    df = read_header(input_fn)[b'foff']
    fchans = read_header(input_fn)[b'nchans']
    return np.arange(fch1, fch1 + fchans * df, df)

def get_ts(input_fn):
    """Gets time values from filterbank file.

    Parameters
    ----------
    input_fn : str
        Name of filterbank file

    Returns
    -------
    ts : ndarray
        Time values
    """
    tsamp = read_header(input_fn)[b'tsamp']
    tchans = get_data(input_fn).shape[0]
    return np.arange(0, tchans*tsamp, tsamp)
