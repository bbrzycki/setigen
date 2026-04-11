from __future__ import annotations

from .recording import _RecordLengthSpec, _resolve_total_obs_num_samples


def _get_block_size(*,
                    num_antennas=1,
                    tchans_per_block=128,
                    num_bits=8,
                    num_pols=2,
                    num_branches=1024,
                    num_chans=64,
                    fftlength=1024,
                    int_factor=4):
    obsnchan = num_chans * num_antennas
    bytes_per_sample = 2 * num_pols * num_bits // 8
    time_samples_per_block = tchans_per_block * fftlength * int_factor
    return time_samples_per_block * obsnchan * bytes_per_sample


def _get_total_obs_num_samples(*,
                               obs_length=None,
                               num_blocks=None,
                               length_mode="obs_length",
                               num_antennas=1,
                               sample_rate=3e9,
                               block_size=134217728,
                               num_bits=8,
                               num_pols=2,
                               num_branches=1024,
                               num_chans=64):
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
