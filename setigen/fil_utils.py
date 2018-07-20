import sys
import os
import errno
import numpy as np
from blimpy import read_header, Waterfall, Filterbank


def maxfreq(input):
    """Return central frequency of the highest-frequency bin in a .fil file.

    """
    if type(input) == str:
        fch1 = read_header(input)[b'fch1']
    elif type(input) == Waterfall:
        fch1 = input.header[b'fch1']
    else:
        sys.exit('Invalid input file!')
    return fch1

def minfreq(input):
    """Return central frequency of the lowest-frequency bin in a .fil file.

    """
    if type(input) == str:
        fch1 = read_header(input)[b'fch1']
        nchans = read_header(input)[b'nchans']
        ch_bandwidth = read_header(input)[b'foff']
    elif type(input) == Waterfall:
        fch1 = input.header[b'fch1']
        nchans = input.header[b'nchans']
        ch_bandwidth = input.header[b'foff']
    else:
        sys.exit('Invalid input file!')
    return fch1 + nchans * ch_bandwidth

def get_data(input):
    """Gets time-frequency data from filterbank file as a 2d NumPy array.

    Parameters
    ----------
    input : str
        Name of filterbank file

    Returns
    -------
    data : ndarray
        Time-frequency data
    """
    if type(input) == str:
        fil = Waterfall(input)
    elif type(input) == Waterfall:
        fil = input
    else:
        sys.exit('Invalid input file!')
    return np.squeeze(fil.data)

def get_fs(input):
    """Gets frequency values from filterbank file.

    Parameters
    ----------
    input : str
        Name of filterbank file

    Returns
    -------
    fs : ndarray
        Frequency values
    """
    if type(input) == str:
        fch1 = read_header(input)[b'fch1']
        df = read_header(input)[b'foff']
        fchans = read_header(input)[b'nchans']
    elif type(input) == Waterfall:
        fch1 = input.header[b'fch1']
        df = input.header[b'foff']
        fchans = input.header[b'nchans']
    else:
        sys.exit('Invalid input file!')
    return np.arange(fch1, fch1 + fchans * df, df)

def get_ts(input):
    """Gets time values from filterbank file.

    Parameters
    ----------
    input : str
        Name of filterbank file

    Returns
    -------
    ts : ndarray
        Time values
    """
    if type(input) == str:
        tsamp = read_header(input)[b'tsamp']
    elif type(input) == Waterfall:
        tsamp = input.header[b'tsamp']
    else:
        sys.exit('Invalid input file!')

    try:
        tchans = get_data(input).shape[0]
    except Exception as e:
        if type(input) == str:
            fch1 = read_header(input)[b'fch1']
            df = read_header(input)[b'foff']
        else:
            fch1 = input.header[b'fch1']
            df = input.header[b'foff']
        fil0 = Waterfall(input, f_start = fch1, f_stop = fch1 + df)
        try:
            tchans = get_data(fil0).shape[0]
        except Exception as e:
            sys.exit('No data in filterbank file!')
    return np.arange(0, tchans * tsamp, tsamp)
