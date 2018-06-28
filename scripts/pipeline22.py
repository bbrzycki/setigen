import numpy as np
from blimpy import read_header, Waterfall, Filterbank

import sys, os

sys.path.append('/datax/users/bryanb/bl-interns/bbrzycki')
import ml_search

sys.path.insert(1,'/datax/users/bryanb/turbo_seti')
from turbo_seti.findoppler.findopp import FinDoppler

f_sample_num = 2**22

input_fn = '/mnt_bls0/datax3/collate/AGBT17B_999_70/spliced_blc00010203040506o7o0111213141516o7o0212223242526o7o031323334353637_guppi_58090_40453_HIP43223_0021.gpuspec.0000.fil'
# /datax/users/bryanb/bl-interns/bbrzycki/jupyter-notebooks/data_dump/1024.fil
# /datax/users/eenriquez/voyager_test/blc07_guppi_57650_67573_Voyager1_0002.gpuspec.0000.fil
output_fn_header = '/datax/users/bryanb/bl-interns/bbrzycki/jupyter-notebooks/data_dump/split_files/%s' % f_sample_num

print('Begin splitting input Filterbank file')

split_fns = ml_search.split_filterbank(input_fn, output_fn_header, f_sample_num)

# This is a hack... necessary to make turbo_seti work?
obs_info = {}
obs_info['pulsar'] = 0  # Bool if pulsar detection.
obs_info['pulsar_found'] = 0  # Bool if pulsar detection.
obs_info['pulsar_dm'] = 0.0  # Pulsar expected DM.
obs_info['pulsar_snr'] = 0.0 # SNR
obs_info['pulsar_stats'] = np.zeros(6)
obs_info['RFI_level'] = 0.0
obs_info['Mean_SEFD'] = 0.0
obs_info['psrflux_Sens'] = 0.0
obs_info['SEFDs_val'] = [0.0]
obs_info['SEFDs_freq'] = [0.0]
obs_info['SEFDs_freq_up'] = [0.0]

for fn in split_fns:
    print('Searching %s' % fn)
    find_seti_event = FinDoppler(fn, max_drift = 10.0, snr = 25.0, out_dir = 'data_dump/split_files/', obs_info=obs_info)
    find_seti_event.search()
    print('Finishing searching %s' % fn)

print('Finished searching everything!')
