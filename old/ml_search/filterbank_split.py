import pandas as pd
#from blimpy import Waterfall
import numpy as np

import os, sys
sys.path.insert(1,'/Users/bbrzycki/Documents/Research/Breakthrough-Listen/Code/blimpy')

import blimpy
from blimpy import read_header, Waterfall

# data_fn = '/Users/bbrzycki/Documents/Research/Breakthrough-Listen/Code/bl-interns/bbrzycki/data/'+'spliced_blc0001020304050607_guppi_58100_78802_OUMUAMUA_0011.gpuspec.0002.fil'

# section = Waterfall(data_fn, f_start=2255, f_stop=2280)
# section.write_to_fil('test2.fil')

def split_filterbank(input_fn, output_fn_header, f_sample_num, f_shift=None):
    """
    Splits filterbank file into smaller filterbank files and write them to disk

    Args:
        input_fn, filename of .fil data
        
    Return:
        List of new files
    """

    fch1 = read_header(input_fn)[b'fch1']
    nchans = read_header(input_fn)[b'nchans']
    ch_bandwidth = read_header(input_fn)[b'foff']

    if f_shift is None:
        f_shift = f_sample_num

    f_start = fch1
    f_stop = fch1 + f_sample_num * ch_bandwidth

    # Iterates down frequencies, starting from highest
    split_fns = []
    index = 0
    while f_stop >= fch1 + nchans * ch_bandwidth:
        output_fn = output_fn_header+'_%04d.fil' % index
        split = Waterfall(input_fn, f_start=f_start, f_stop=f_stop)
        split.write_to_fil(output_fn)
        f_start += f_shift * ch_bandwidth
        f_stop += f_shift * ch_bandwidth
        index += 1
        split_fns.append(output_fn)
        print('Saved %s' % output_fn)
    return split_fns

# split_filterbank(data_fn, 262144)
