import numpy as np
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from blimpy import read_header, Waterfall, Filterbank

import matplotlib.pyplot as plt

import sys, os, glob
sys.path.append("../")
import setigen as stg

######

max_drift_rate = 10 * 1e-6 # Hz/s * 1e-6 MHz/Hz
size_lim = 1e9 # bytes

#
# f_sample_num = 2**20

input_fn = '/mnt_bls0/datax3/collate/AGBT17B_999_70/spliced_blc00010203040506o7o0111213141516o7o0212223242526o7o031323334353637_guppi_58090_40453_HIP43223_0021.gpuspec.0000.fil'
output_dir = '/datax/users/bryanb/data/'

ts = stg.get_ts(input_fn)
tsamp = ts[1] - ts[0]
fs = stg.get_fs(input_fn)
df = fs[1] - fs[0]

index_range = int(np.floor(np.abs(max_drift_rate * (len(ts) * tsamp) / df)))
print(index_range)

fchans = len(fs)
print(fchans)
min_range = index_range
max_range = index_range
while fchans % min_range != 0:
    min_range -= 1
while fchans % max_range != 0:
    max_range += 1
print('Min: %s' % min_range)
print('Max: %s' % max_range)

if index_range - min_range < max_range - index_range:
    index_range = min_range
else:
    index_range = max_range



print('Begin splitting input Filterbank file')

split_fns = stg.split_fil(input_fn, output_dir, f_sample_num)

######

# create a file handler
handler = logging.FileHandler('10.log')
handler.setLevel(logging.INFO)

# create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(handler)

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
    # print('Searching 10: %s' % fn)
    logger.info('Searching 10: %s' % fn)
    find_seti_event = FinDoppler(fn, max_drift = 10.0, snr = 25.0, out_dir = output_dir+'search_10/', obs_info=obs_info)
    find_seti_event.search()
    # print('Finishing searching %s' % fn)
    logger.info('Finishing searching %s' % fn)

print('Finished searching everything with turbo_seti!')

######

# print('Starting dataframe search')
logger.info('Starting dataframe search')
# Sort files with and without Doppler drifted hits
no_hits = []
hits_no_doppler = []
hits_with_doppler = []

df = pd.DataFrame()
for fn in split_fns:
    head, tail = os.path.split(fn)

    csv = head + 'search_10/' + tail[:-4] + '.dat'

    # print('On %s' % csv)
    names = ['Top_Hit_#', 'Drift_Rate', 'SNR', 'Uncorrected_Frequency', 'Corrected_Frequency', 'Index', 'freq_start', 'freq_end', 'SEFD', 'SEFD_freq', 'Coarse_Channel_Number', 'Full_number_of_hits']
    dataframe = pd.read_csv(csv, delim_whitespace=True, comment='#', names=names)

    if len(dataframe) == 0:
        no_hits.append(i)
    elif all(dataframe['Drift_Rate'] == 0.0):
        hits_no_doppler.append(i)
    else:
        hits_with_doppler.append(i)
print('# no hits: %s' % len(no_hits))
print('# hits no doppler: %s' % len(hits_no_doppler))
print('# hits with doppler: %s' % len(hits_with_doppler))

with open(output_dir+'search_10/no_hits.txt', 'w') as no_hits_file:
    for index in no_hits:
        no_hits_file.write('%s\n' % index)
with open(output_dir+'search_10/hits_no_doppler.txt', 'w') as hits_no_doppler_file:
    for index in hits_no_doppler:
        hits_no_doppler_file.write('%s\n' % index)
with open(output_dir+'search_10/hits_with_doppler.txt', 'w') as hits_with_doppler_file:
    for index in hits_with_doppler:
        hits_with_doppler_file.write('%s\n' % index)

# print('Finished writing out to file')
logger.info('Finished writing out to file')



######

# create a file handler
handler = logging.FileHandler('100.log')
handler.setLevel(logging.INFO)

# create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(handler)

for fn in split_fns:
    # print('Searching 100: %s' % fn)
    logger.info('Searching 100: %s' % fn)
    find_seti_event = FinDoppler(fn, max_drift = 100.0, snr = 25.0, out_dir = output_dir+'search_100/', obs_info=obs_info)
    find_seti_event.search()
    # print('Finishing searching %s' % fn)
    logger.info('Finishing searching %s' % fn)

print('Finished searching everything with turbo_seti!')
