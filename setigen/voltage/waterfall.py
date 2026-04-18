import os

GPU_FLAG = os.getenv('SETIGEN_ENABLE_GPU', '0')
if GPU_FLAG == '1':
    try:
        import cupy as xp
    except ImportError:
        import numpy as xp
else:
    import numpy as xp
    
import numpy as np
import time

from ._reduction.channelize import _channelize_block
from ._reduction.decoder import _decode_raw_block
from ._reduction.input import _resolve_raw_input


def get_pfb_waterfall(pfb_voltages_x, pfb_voltages_y=None, fftlength=256, int_factor=1):
    """
    Perform fine channelization on input complex voltages after filterbank,
    for single or dual polarizations. 
    
    Parameters
    ----------
    pfb_voltages_x : array
        Complex voltages in first polarization, of shape (time_samples, num_chans)
    pfb_voltages_y : array, optional
        Complex voltages in second polarization, of shape (time_samples, num_chans)
    fftlength : int
        FFT length to be used in fine channelization
    int_factor : int, optional
        Integration factor to be used in fine channelization
    
    Returns
    -------
    XX_psd : array
        Finely channelized voltages
    """
    
    XX_psd = xp.zeros((pfb_voltages_x.shape[1], pfb_voltages_x.shape[0] // fftlength, fftlength))
    
    pfb_voltages_list = [pfb_voltages_x]
    if pfb_voltages_y is not None:
        pfb_voltages_list.append(pfb_voltages_y)
        
    for pfb_voltages in pfb_voltages_list:
        X_samples = pfb_voltages.T
        X_samples = X_samples[:, :(X_samples.shape[1] // fftlength) * fftlength]
        X_samples = X_samples.reshape((X_samples.shape[0], X_samples.shape[1] // fftlength, fftlength))
        XX = xp.fft.fft(X_samples, fftlength, axis=2) / fftlength**0.5
        XX = xp.fft.fftshift(XX, axes=2)
        XX_psd += xp.abs(XX)**2 

    XX_psd = xp.concatenate(XX_psd, axis=1)
    
    # Integrate over time, trimming if necessary
    XX_psd = XX_psd[:(XX_psd.shape[0] // int_factor) * int_factor]
    XX_psd = XX_psd.reshape(XX_psd.shape[0] // int_factor, int_factor, XX_psd.shape[1])
    XX_psd = XX_psd.sum(axis=1)
    
    return XX_psd


def get_waterfall_from_raw(raw_filename, block_size, num_chans, int_factor=1, fftlength=256):
    """ 
    Produces waterfall data array from the first block of a dual-polarized, 8 bit RAW file. Lightweight 
    function mainly for testing. 
    
    Parameters
    ----------
    raw_filename : str
        Filename of GUPPI RAW file
    block_size : int
        Number of bytes in a data block
    num_chans : int
        Number of coarse channels saved in RAW file
    fftlength : int
        FFT length to be used in fine channelization
    int_factor : int, optional
        Integration factor to be used in fine channelization
    
    Returns
    -------
    XX_psd : array
        Finely channelized voltages
    """
    input_spec = _resolve_raw_input(raw_filename)
    if input_spec.block_size != block_size:
        raise ValueError(f"Provided block_size={block_size} does not match RAW header BLOCSIZE={input_spec.block_size}.")
    if input_spec.num_chans != num_chans:
        raise ValueError(f"Provided num_chans={num_chans} does not match RAW header OBSNCHAN={input_spec.num_chans}.")

    with open(input_spec.files[0], "rb") as handle:
        handle.read(input_spec.header_size)
        chunk = handle.read(block_size)

    voltages = _decode_raw_block(chunk,
                                 num_bits=input_spec.num_bits,
                                 num_pols=input_spec.num_pols,
                                 num_chans=input_spec.num_chans,
                                 start_chan=0,
                                 num_selected_chans=input_spec.num_chans)
    reduced = _channelize_block(voltages,
                                fftlength=fftlength,
                                integration_factor=int_factor,
                                pol_mode=1,
                                backend="auto")
    return reduced[:, 0, :]
