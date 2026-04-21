from __future__ import annotations

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

from ._reduction.channelize import _channelize_block
from ._reduction.decoder import _decode_raw_block
from ._reduction.input import _resolve_raw_input


def get_pfb_waterfall(pfb_voltages_x: np.ndarray,
                      pfb_voltages_y: np.ndarray | None = None,
                      fftlength: int = 256,
                      int_factor: int = 1) -> np.ndarray:
    """Fine-channelize already-channelized complex voltages.

    Args:
        pfb_voltages_x: Complex voltages for the first polarization with shape
            ``(time_samples, num_chans)``.
        pfb_voltages_y: Optional complex voltages for the second polarization
            with the same shape as ``pfb_voltages_x``.
        fftlength: Fine-channel FFT length.
        int_factor: Time integration factor after fine channelization.

    Returns:
        Integrated power waterfall with shape ``(tchans, fchans)``.
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


def get_waterfall_from_raw(raw_filename: str,
                           block_size: int,
                           num_chans: int,
                           int_factor: int = 1,
                           fftlength: int = 256) -> np.ndarray:
    """Reduce the first RAW block into a Stokes-I waterfall array.

    This helper is intentionally lightweight and primarily intended for tests
    and quick inspection of a single RAW block.

    Args:
        raw_filename: GUPPI RAW file path or RAW stem.
        block_size: Expected number of bytes in a RAW data block.
        num_chans: Expected number of coarse channels in the RAW file.
        int_factor: Time integration factor after fine channelization.
        fftlength: Fine-channel FFT length.

    Returns:
        Stokes-I waterfall for the first RAW block.

    Raises:
        ValueError: If the provided block metadata does not match the RAW
            header values.
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
