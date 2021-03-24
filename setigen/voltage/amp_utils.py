import numpy as np


def get_unit_drift_rate(raw_voltage_backend,
                        fftlength=1048576,
                        int_factor=1):
    """
    Calculate drift rate corresponding to a 1x1 pixel shift in the final data product.
    """
    df = raw_voltage_backend.chan_bw / fftlength
    dt = raw_voltage_backend.tbin * fftlength * int_factor
    return df / dt


def get_intensity(snr, 
                  raw_voltage_backend,
                  obs_length=None, 
                  num_blocks=None,
                  length_mode='obs_length',
                  fftlength=1048576):
    if length_mode == 'obs_length':
        if obs_length is None:
            raise ValueError("Value not given for 'obs_length'.")
        num_blocks = raw_voltage_backend.get_num_blocks(obs_length)
    elif length_mode == 'num_blocks':
        if num_blocks is None:
            raise ValueError("Value not given for 'num_blocks'.")
        pass
    else:
        raise ValueError("Invalid option given for 'length_mode'.")
            
    # Get amplitude required for cosine signal to get required SNR
    int_factor = 1 # level has no dependence on integration factor
    dt = raw_voltage_backend.tbin * fftlength * int_factor
    tchans = int(raw_voltage_backend.time_per_block * num_blocks / dt)
    
    chi_df = 2 * raw_voltage_backend.num_pols * int_factor
#     main_mean = (raw_voltage_backend.requantizer.target_sigma)**2 * chi_df * raw_voltage_backend.filterbank.max_mean_ratio
    
    I_per_SNR = np.sqrt(2 / chi_df) / tchans**0.5
 
    signal_level = 1 / (raw_voltage_backend.num_branches * fftlength / 4)**0.5 * (snr * I_per_SNR)**0.5 
    return signal_level


def get_leakage_factor(f_start,
                       raw_voltage_backend,
                       fftlength):
    """
    Get factor to scale up signal amplitude from spectral leakage based on the 
    position of a signal in a fine channel
    """ 
    spectral_bin_frac = np.modf((f_start - raw_voltage_backend.fch1) / (raw_voltage_backend.chan_bw / fftlength))[0]
    spectral_bin_frac = np.min([spectral_bin_frac, 1 - spectral_bin_frac])
    return 1 / np.sinc(spectral_bin_frac)


def get_total_obs_num_samples(obs_length=None, 
                              num_blocks=None, 
                              num_antennas=1,
                              sample_rate=3e9,
                              block_size=134217728,
                              num_bits=8,
                              num_pols=2,
                              num_branches=1024,
                              num_chans=64, 
                              length_mode='obs_length'):
    """
    length_mode can be 'obs_length' or 'num_blocks'
    """
    tbin = num_branches / sample_rate
    chan_bw = 1 / tbin
    bytes_per_sample = 2 * num_pols * num_bits / 8
    if length_mode == 'obs_length':
        if obs_length is None:
            raise ValueError("Value not given for 'obs_length'.")
        num_blocks = int(obs_length * chan_bw * num_antennas * num_chans * bytes_per_sample / block_size)
    elif length_mode == 'num_blocks':
        if num_blocks is None:
            raise ValueError("Value not given for 'num_blocks'.")
        pass
    else:
        raise ValueError("Invalid option given for 'length_mode'.")
    return num_blocks * block_size / (num_antennas * num_chans * bytes_per_sample) * num_branches

