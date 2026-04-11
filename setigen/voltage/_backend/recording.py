from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


class _LengthMode(str, Enum):
    OBS_LENGTH = "obs_length"
    NUM_BLOCKS = "num_blocks"


def _coerce_length_mode(length_mode):
    if isinstance(length_mode, _LengthMode):
        return length_mode
    try:
        return _LengthMode(length_mode)
    except ValueError as exc:
        raise ValueError("Invalid option given for 'length_mode'.") from exc


@dataclass(frozen=True)
class _RecordLengthSpec:
    mode: _LengthMode = _LengthMode.OBS_LENGTH
    obs_length: float | None = None
    num_blocks: int | None = None

    @classmethod
    def from_values(cls,
                    obs_length=None,
                    num_blocks=None,
                    length_mode="obs_length"):
        return cls(mode=_coerce_length_mode(length_mode),
                   obs_length=obs_length,
                   num_blocks=num_blocks)


@dataclass(frozen=True)
class _RecordConfig:
    length: _RecordLengthSpec = field(default_factory=_RecordLengthSpec)
    header_dict: dict = field(default_factory=dict)
    digitize: bool = True
    load_template: bool = True
    verbose: bool = True

    @classmethod
    def from_values(cls,
                    obs_length=None,
                    num_blocks=None,
                    length_mode="obs_length",
                    header_dict=None,
                    digitize=True,
                    load_template=True,
                    verbose=True):
        return cls(length=_RecordLengthSpec.from_values(obs_length=obs_length,
                                                        num_blocks=num_blocks,
                                                        length_mode=length_mode),
                   header_dict={} if header_dict is None else dict(header_dict),
                   digitize=digitize,
                   load_template=load_template,
                   verbose=verbose)


def _resolve_num_blocks(length_spec,
                        *,
                        get_num_blocks: Callable[[float], int],
                        fallback_num_blocks=None):
    if length_spec.mode is _LengthMode.OBS_LENGTH:
        if length_spec.obs_length is None:
            if fallback_num_blocks is not None:
                return fallback_num_blocks
            raise ValueError("Value not given for 'obs_length'.")
        return get_num_blocks(length_spec.obs_length)

    if length_spec.num_blocks is None:
        if fallback_num_blocks is not None:
            return fallback_num_blocks
        raise ValueError("Value not given for 'num_blocks'.")
    return length_spec.num_blocks


def _resolve_total_obs_num_samples(length_spec,
                                   *,
                                   num_antennas=1,
                                   sample_rate=3e9,
                                   block_size=134217728,
                                   num_bits=8,
                                   num_pols=2,
                                   num_branches=1024,
                                   num_chans=64):
    tbin = num_branches / sample_rate
    chan_bw = 1 / tbin
    bytes_per_sample = 2 * num_pols * num_bits / 8
    num_blocks = _resolve_num_blocks(
        length_spec,
        get_num_blocks=lambda obs_length: int(obs_length
                                              * chan_bw
                                              * num_antennas
                                              * num_chans
                                              * bytes_per_sample
                                              / block_size),
    )
    samples_per_block = int(block_size / (num_antennas * num_chans * bytes_per_sample))
    return num_blocks * samples_per_block * num_branches
