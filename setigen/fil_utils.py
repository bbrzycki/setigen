import sys
import numpy as np
from blimpy import read_header, Waterfall


def maxfreq(fil):
    """
    Returns central frequency of the highest-frequency bin in a .fil file.

    Parameters
    ----------
    fil : str or Waterfall
        Name of filterbank file or Waterfall object

    Returns
    -------
    fmax : float
        Maximum frequency in data
    """
    if type(fil) == str:
        fch1 = read_header(fil)[b'fch1']
    elif type(fil) == Waterfall:
        fch1 = fil.header[b'fch1']
    else:
        sys.exit('Invalid fil file!')

    return fch1


def minfreq(fil):
    """
    Returns central frequency of the lowest-frequency bin in a .fil file.

    Parameters
    ----------
    fil : str or Waterfall
        Name of filterbank file or Waterfall object

    Returns
    -------
    fmin : float
        Minimum frequency in data
    """
    if type(fil) == str:
        fch1 = read_header(fil)[b'fch1']
        nchans = read_header(fil)[b'nchans']
        ch_bandwidth = read_header(fil)[b'foff']
    elif type(fil) == Waterfall:
        fch1 = fil.header[b'fch1']
        nchans = fil.header[b'nchans']
        ch_bandwidth = fil.header[b'foff']
    else:
        sys.exit('Invalid fil file!')

    return fch1 + nchans * ch_bandwidth


def get_data(fil, db=False):
    """
    Gets time-frequency data from filterbank file as a 2d NumPy array.

    Note: when multiple Stokes parameters are supported, this will have to
    be expanded.

    Parameters
    ----------
    fil : str or Waterfall
        Name of filterbank file or Waterfall object

    Returns
    -------
    data : ndarray
        Time-frequency data
    """
    if type(fil) == str:
        fil = Waterfall(fil)
    elif type(fil) == Waterfall:
        pass
    else:
        sys.exit('Invalid fil file!')

    if db:
        return 10 * np.log10(np.squeeze(fil.data))

    return np.squeeze(fil.data)


def get_fs(fil):
    """
    Gets frequency values from filterbank file.

    Parameters
    ----------
    fil : str or Waterfall
        Name of filterbank file or Waterfall object

    Returns
    -------
    fs : ndarray
        Frequency values
    """
    if type(fil) == str:
        fch1 = read_header(fil)[b'fch1']
        df = read_header(fil)[b'foff']
        fchans = read_header(fil)[b'nchans']
    elif type(fil) == Waterfall:
        fch1 = fil.header[b'fch1']
        df = fil.header[b'foff']
        fchans = fil.header[b'nchans']
    else:
        sys.exit('Invalid fil file!')

    return np.arange(fch1, fch1 + fchans * df, df)


def get_ts(fil):
    """
    Gets time values from filterbank file.

    Parameters
    ----------
    fil : str or Waterfall
        Name of filterbank file or Waterfall object

    Returns
    -------
    ts : ndarray
        Time values
    """
    if type(fil) == str:
        tsamp = read_header(fil)[b'tsamp']
    elif type(fil) == Waterfall:
        tsamp = fil.header[b'tsamp']
    else:
        sys.exit('Invalid fil file!')

    if type(fil) == str:
        fch1 = read_header(fil)[b'fch1']
        df = read_header(fil)[b'foff']
    else:
        fch1 = fil.header[b'fch1']
        df = fil.header[b'foff']

    fil0 = Waterfall(fil, f_start=fch1, f_stop=fch1 + df)

    try:
        tchans = get_data(fil0).shape[0]
    except Exception as e:
        sys.exit('No data in filterbank file!')

    return np.arange(0, tchans * tsamp, tsamp)
