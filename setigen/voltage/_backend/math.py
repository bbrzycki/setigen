from __future__ import annotations

from .recording import _LengthMode, _RecordLengthSpec, _resolve_total_obs_num_samples


def _get_block_size(*,
                    num_antennas: int = 1,
                    tchans_per_block: int = 128,
                    num_bits: int = 8,
                    num_pols: int = 2,
                    num_branches: int = 1024,
                    num_chans: int = 64,
                    fftlength: int = 1024,
                    int_factor: int = 4) -> int:
    """Calculate the RAW block size for a backend configuration.

    Args:
        num_antennas: Number of antennas contributing voltages.
        tchans_per_block: Number of integrated time channels per RAW block.
        num_bits: Bit depth of each complex voltage component.
        num_pols: Number of polarizations.
        num_branches: Number of PFB branches.
        num_chans: Number of coarse channels saved per antenna.
        fftlength: Fine-channel FFT length.
        int_factor: Time integration factor after fine channelization.

    Returns:
        RAW block size in bytes.
    """
    obsnchan = num_chans * num_antennas
    bytes_per_sample = 2 * num_pols * num_bits // 8
    time_samples_per_block = tchans_per_block * fftlength * int_factor
    return time_samples_per_block * obsnchan * bytes_per_sample


def _get_total_obs_num_samples(*,
                               obs_length: float | None = None,
                               num_blocks: int | None = None,
                               length_mode: str | _LengthMode = "obs_length",
                               num_antennas: int = 1,
                               sample_rate: float = 3e9,
                               block_size: int = 134217728,
                               num_bits: int = 8,
                               num_pols: int = 2,
                               num_branches: int = 1024,
                               num_chans: int = 64) -> int:
    """Get the total number of voltage samples in an observation.

    Args:
        obs_length: Observation length in seconds when using
            ``length_mode="obs_length"``.
        num_blocks: Number of RAW blocks when using
            ``length_mode="num_blocks"``.
        length_mode: Observation-length interpretation mode.
        num_antennas: Number of antennas contributing voltages.
        sample_rate: Complex-voltage sample rate in Hz.
        block_size: RAW block size in bytes.
        num_bits: Bit depth of each complex voltage component.
        num_pols: Number of polarizations.
        num_branches: Number of PFB branches.
        num_chans: Number of coarse channels saved per antenna.

    Returns:
        Total number of complex voltage samples across the observation.
    """
    return _resolve_total_obs_num_samples(
        _RecordLengthSpec.from_values(obs_length=obs_length,
                                      num_blocks=num_blocks,
                                      length_mode=length_mode),
        num_antennas=num_antennas,
        sample_rate=sample_rate,
        block_size=block_size,
        num_bits=num_bits,
        num_pols=num_pols,
        num_branches=num_branches,
        num_chans=num_chans,
    )
