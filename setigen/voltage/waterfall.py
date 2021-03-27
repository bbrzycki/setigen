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
import scipy.signal
import time


def get_pfb_waterfall(pfb_voltages_x, pfb_voltages_y=None, int_factor=1, fftlength=256):
    """
    Perform fine channelization on input complex voltages after filterbank,
    for single or dual polarizations. 
    
    int_factor specifies the number of time samples to integrate.
    
    Shape of pfb_voltages is (time_samples, num_channels).
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
    # produces waterfall from a raw file (only the first block), 2pol, 8bit
    with open(raw_filename, "rb") as f:
        i = 1
        chunk = f.read(80)
        while f"{'END':<80}".encode() not in chunk:
            chunk = f.read(80)
            i += 1
        # Skip zero padding
        chunk = f.read((512 - (80 * i % 512)))
        # Read data
        chunk = f.read(block_size)
        
    rawbuffer = np.frombuffer(chunk, dtype=xp.int8).reshape((num_chans, -1))
    rawbuffer_x = rawbuffer[:, 0::4] + rawbuffer[:, 1::4] * 1j
    rawbuffer_y = rawbuffer[:, 2::4] + rawbuffer[:, 3::4] * 1j    
    return get_pfb_waterfall(rawbuffer_x.T, rawbuffer_y.T, int_factor, fftlength)
