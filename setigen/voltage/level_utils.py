import numpy as np

from setigen.voltage._backend.recording import _RecordLengthSpec, _resolve_num_blocks


def get_unit_drift_rate(raw_voltage_backend,
                        fftlength,
                        int_factor=1):
    """
    Calculate drift rate corresponding to a 1x1 pixel shift in the final data product.
    This is equivalent to dividing the fine channelized frequency resolution with the
    time resolution.
    
    Parameters
    ----------
    raw_voltage_backend : RawVoltageBackend
        Backend object to infer observation parameters
    fftlength : int
        FFT length to be used in fine channelization
    int_factor : int, optional
        Integration factor to be used in fine channelization
    
    Returns
    -------
    unit_drift_rate : float
        Drift rate in Hz / s
    """
    df = raw_voltage_backend.chan_bw / fftlength
    dt = raw_voltage_backend.tbin * fftlength * int_factor
    return df / dt


def get_level(snr, 
              raw_voltage_backend,
              fftlength,
              obs_length=None, 
              num_blocks=None,
              length_mode='obs_length'):
    """
    Calculate required signal level as a function of desired SNR, assuming initial noise 
    variance of 1. This is calculated for a single polarization. This further assumes the signal
    is non-drifting and centered on a finely channelized bin. 
    
    Parameters
    ----------
    snr : float
        Signal-to-noise ratio (SNR)
    raw_voltage_backend : RawVoltageBackend
        Backend object to infer observation parameters
    fftlength : int, optional
        FFT length to be used in fine channelization
    obs_length : float, optional
        Length of observation in seconds, if in 'obs_length' mode
    num_blocks : int, optional
        Number of data blocks to record, if in 'num_blocks' mode
    length_mode : str, optional
        Mode for specifying length of observation, either 'obs_length' in seconds or 'num_blocks' in data blocks
    
    Returns
    -------
    level : float
        Level, or amplitude, for a real voltage cosine signal
    """
    num_blocks = _resolve_num_blocks(
        _RecordLengthSpec.from_values(obs_length=obs_length,
                                      num_blocks=num_blocks,
                                      length_mode=length_mode),
        get_num_blocks=raw_voltage_backend.get_num_blocks,
    )
            
    # Get amplitude required for cosine signal to get required SNR
    int_factor = 1 # level has no dependence on integration factor
    dt = raw_voltage_backend.tbin * fftlength * int_factor
    tchans = int(raw_voltage_backend.time_per_block * num_blocks / dt)
    
    chi_df = 2 * raw_voltage_backend.num_pols * int_factor
    # main_mean = (raw_voltage_backend.requantizer.target_sigma)**2 * chi_df * raw_voltage_backend.filterbank.max_mean_ratio
    
    I_per_SNR = np.sqrt(2 / chi_df) / tchans**0.5
 
    signal_level = 1 / (raw_voltage_backend.num_branches * fftlength / 4)**0.5 * (snr * I_per_SNR)**0.5 
    return signal_level


def get_leakage_factor(f_start,
                       raw_voltage_backend,
                       fftlength):
    """
    Get factor to scale up signal amplitude from spectral leakage based on the 
    position of a signal in a fine channel. This calculates an inverse normalized 
    sinc value based on the position of the signal with respect to finely channelized bins.
    Since intensity goes as voltage squared, this gives a scaling proportional to 1/sinc^2 
    in finely channelized data products; this is the standard fine channel spectral response.
    
    Parameters
    ----------
    f_start : float
        Signal frequency, in Hz
    raw_voltage_backend : RawVoltageBackend
        Backend object to infer observation parameters
    fftlength : int, optional
        FFT length to be used in fine channelization
    
    Returns
    -------
    leakage_factor : float
        Factor to multiply to signal level / amplitude
    """ 
    spectral_bin_frac = np.modf((f_start - raw_voltage_backend.fch1) / (raw_voltage_backend.chan_bw / fftlength))[0]
    spectral_bin_frac = np.min([spectral_bin_frac, 1 - spectral_bin_frac])
    return 1 / np.sinc(spectral_bin_frac)
