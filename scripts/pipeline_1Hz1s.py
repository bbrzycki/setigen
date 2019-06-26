import numpy as np
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from blimpy import read_header, Waterfall, Filterbank

import matplotlib.pyplot as plt

import sys, os, glob, errno
sys.path.append("../")
import setigen as stg

# import edited version of turbo_seti
sys.path.insert(1,'../../turbo_seti')
from turbo_seti.findoppler.findopp import FinDoppler

######

f_sample_num = 2**20

input_fn = '/datax/scratch/bbrzycki/data/observations/blc15_guppi_58362_03714_DIAG_SGR_B2_0069.rawspec.0000.fil'
output_dir = '/datax/scratch/bbrzycki/data/observations/turbo_hits/'

try:
    os.makedirs(output_dir)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

try:
    os.makedirs(output_dir+'search_10/')
except:
    pass
try:
    os.makedirs(output_dir+'search_100/')
except:
    pass

print('Begin splitting input Filterbank file')

# split_fns = stg.split_fil(input_fn, output_dir, f_sample_num)
split_fns = glob.glob('/datax/scratch/bbrzycki/data/observations/turbo_hits/*.fil')




######

# create a file handler
handler = logging.FileHandler('/datax/scratch/bbrzycki/data/observations/turbo_hits/10.log')
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
    
    csv = output_dir + 'search_10/' + tail[:-4] + '.dat'
    
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
handler = logging.FileHandler('/datax/scratch/bbrzycki/data/observations/turbo_hits/100.log')
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

######

#print('Starting dataframe search')
logger.info('Starting dataframe search')
# Sort files with and without Doppler drifted hits
no_hits = []
hits_no_doppler = []
hits_with_doppler = []

df = pd.DataFrame()
for fn in split_fns:
    head, tail = os.path.split(fn)
    
    csv = output_dir + 'search_100/' + tail[:-4] + '.dat'
    
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

with open(output_dir+'search_100/no_hits.txt', 'w') as no_hits_file:
    for index in no_hits:
        no_hits_file.write('%s\n' % index)
with open(output_dir+'search_100/hits_no_doppler.txt', 'w') as hits_no_doppler_file:
    for index in hits_no_doppler:
        hits_no_doppler_file.write('%s\n' % index)
with open(output_dir+'search_100/hits_with_doppler.txt', 'w') as hits_with_doppler_file:
    for index in hits_with_doppler:
        hits_with_doppler_file.write('%s\n' % index)
        
# print('Finished writing out to file')
logger.info('Finished writing out to file')
