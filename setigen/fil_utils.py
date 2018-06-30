import sys
import os
import errno
import numpy as np
from blimpy import read_header, Waterfall, Filterbank

def maxfreq(input_fn):
    """
    Return central frequency of the highest-frequency bin in a .fil file.
    """
    return read_header(input_fn)[b'fch1']

def minfreq(input_fn):
    """
    Return central frequency of the lowest-frequency bin in a .fil file
    """
    fch1 = read_header(input_fn)[b'fch1']
    nchans = read_header(input_fn)[b'nchans']
    ch_bandwidth = read_header(input_fn)[b'foff']
    return fch1 + nchans * ch_bandwidth

def get_data(input_fn):
    """
    Gets time-frequency data from filterbank file as a 2d NumPy array

    Args:
        input_fn, name of filterbank file

    Return:
        NumPy array with time-frequency data
    """
    fil = Waterfall(input_fn)
    return np.squeeze(fil.data)
