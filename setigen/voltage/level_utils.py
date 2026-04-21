from __future__ import annotations

import numpy as np

from .backend import RawVoltageBackend
from setigen.voltage._backend.recording import _RecordLengthSpec, _resolve_num_blocks


def get_unit_drift_rate(raw_voltage_backend: RawVoltageBackend,
                        fftlength: int,
                        int_factor: int = 1) -> float:
    """Get the drift rate corresponding to a one-pixel shift.

    Args:
        raw_voltage_backend: Backend object used to infer observation
            parameters.
        fftlength: Fine-channel FFT length.
        int_factor: Time integration factor used after fine channelization.

    Returns:
        Drift rate in Hz/s corresponding to a 1x1 pixel shift.
    """
    df = raw_voltage_backend.chan_bw / fftlength
    dt = raw_voltage_backend.tbin * fftlength * int_factor
    return df / dt


def get_level(snr: float,
              raw_voltage_backend: RawVoltageBackend,
              fftlength: int,
              obs_length: float | None = None,
              num_blocks: int | None = None,
              length_mode: str = 'obs_length') -> float:
    """Calculate the cosine voltage level needed to reach a target SNR.

    This calibration assumes unit input noise variance, a single-polarization
    signal, no drift, and a tone centered exactly on a fine FFT bin.

    Args:
        snr: Desired signal-to-noise ratio.
        raw_voltage_backend: Backend object used to infer observation
            parameters.
        fftlength: Fine-channel FFT length.
        obs_length: Observation length in seconds when using
            ``length_mode="obs_length"``.
        num_blocks: Number of RAW blocks when using
            ``length_mode="num_blocks"``.
        length_mode: Observation-length interpretation mode.

    Returns:
        Signal amplitude for a real cosine voltage injection.

    Raises:
        ValueError: If the requested observation length and FFT length produce
            no complete fine-channelized spectra.
    """
    num_blocks = _resolve_num_blocks(
        _RecordLengthSpec.from_values(obs_length=obs_length,
                                      num_blocks=num_blocks,
                                      length_mode=length_mode),
        get_num_blocks=raw_voltage_backend.get_num_blocks,
    )
            
    # Get amplitude required for cosine signal to get required SNR
    int_factor = 1  # level has no dependence on integration factor
    dt = raw_voltage_backend.tbin * fftlength * int_factor
    tchans = int(raw_voltage_backend.time_per_block * num_blocks / dt)
    if tchans <= 0:
        raise ValueError(
            "Requested observation length and fftlength produce no complete fine-channelized spectra."
        )
    
    chi_df = 2 * raw_voltage_backend.num_pols * int_factor
    # main_mean = (raw_voltage_backend.requantizer.target_sigma)**2 * chi_df * raw_voltage_backend.filterbank.max_mean_ratio
    
    I_per_SNR = np.sqrt(2 / chi_df) / tchans**0.5
 
    signal_level = 1 / (raw_voltage_backend.num_branches * fftlength / 4)**0.5 * (snr * I_per_SNR)**0.5
    return signal_level


def get_leakage_factor(f_start: float,
                       raw_voltage_backend: RawVoltageBackend,
                       fftlength: int) -> float:
    """Get the fine-channel spectral-leakage correction factor.

    Args:
        f_start: Signal frequency in Hz.
        raw_voltage_backend: Backend object used to infer observation
            parameters.
        fftlength: Fine-channel FFT length.

    Returns:
        Multiplicative correction factor for signal amplitude.
    """
    spectral_bin_frac = np.modf((f_start - raw_voltage_backend.fch1) / (raw_voltage_backend.chan_bw / fftlength))[0]
    spectral_bin_frac = np.min([spectral_bin_frac, 1 - spectral_bin_frac])
    return 1 / np.sinc(spectral_bin_frac)
