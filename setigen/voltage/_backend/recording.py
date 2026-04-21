from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class _LengthMode(str, Enum):
    """Supported ways to specify observation length."""

    OBS_LENGTH = "obs_length"
    NUM_BLOCKS = "num_blocks"


def _coerce_length_mode(length_mode: str | _LengthMode) -> _LengthMode:
    """Normalize a user-supplied length mode value.

    Args:
        length_mode: Raw length-mode value from the caller.

    Returns:
        Normalized enum value.

    Raises:
        ValueError: If the mode is not supported.
    """
    if isinstance(length_mode, _LengthMode):
        return length_mode
    try:
        return _LengthMode(length_mode)
    except ValueError as exc:
        raise ValueError("Invalid option given for 'length_mode'.") from exc


@dataclass(frozen=True)
class _RecordLengthSpec:
    """Normalized observation-length request for RAW recording."""

    mode: _LengthMode = _LengthMode.OBS_LENGTH
    obs_length: float | None = None
    num_blocks: int | None = None

    @classmethod
    def from_values(
        cls,
        obs_length: float | None = None,
        num_blocks: int | None = None,
        length_mode: str | _LengthMode = "obs_length",
    ) -> "_RecordLengthSpec":
        """Build a normalized observation-length request from raw user inputs.

        Args:
            obs_length: Observation length in seconds.
            num_blocks: Explicit number of RAW blocks.
            length_mode: Strategy for interpreting the supplied length values.

        Returns:
            Normalized record-length specification.
        """
        return cls(
            mode=_coerce_length_mode(length_mode),
            obs_length=obs_length,
            num_blocks=num_blocks,
        )


@dataclass(frozen=True)
class _RecordConfig:
    """Full normalized recording configuration for one `record()` call."""

    length: _RecordLengthSpec = field(default_factory=_RecordLengthSpec)
    header_dict: dict[str, Any] = field(default_factory=dict)
    digitize: bool = True
    load_template: bool = True
    verbose: bool = True

    @classmethod
    def from_values(
        cls,
        obs_length: float | None = None,
        num_blocks: int | None = None,
        length_mode: str | _LengthMode = "obs_length",
        header_dict: dict[str, Any] | None = None,
        digitize: bool = True,
        load_template: bool = True,
        verbose: bool = True,
    ) -> "_RecordConfig":
        """Build a normalized recording configuration from public API inputs.

        Args:
            obs_length: Observation length in seconds.
            num_blocks: Explicit number of RAW blocks.
            length_mode: Strategy for interpreting the supplied length values.
            header_dict: Optional header overrides.
            digitize: Whether to digitize input voltages before the PFB.
            load_template: Whether to merge the built-in RAW header template.
            verbose: Whether to emit progress output.

        Returns:
            Normalized recording configuration.
        """
        return cls(
            length=_RecordLengthSpec.from_values(
                obs_length=obs_length,
                num_blocks=num_blocks,
                length_mode=length_mode,
            ),
            header_dict={} if header_dict is None else dict(header_dict),
            digitize=digitize,
            load_template=load_template,
            verbose=verbose,
        )


def _resolve_num_blocks(
    length_spec: _RecordLengthSpec,
    *,
    get_num_blocks: Callable[[float], int],
    fallback_num_blocks: int | None = None,
) -> int:
    """Resolve an observation-length request into an integer number of RAW blocks.

    Args:
        length_spec: Normalized record-length specification.
        get_num_blocks: Callback that converts seconds into RAW block count.
        fallback_num_blocks: Optional fallback when no explicit length is supplied.

    Returns:
        Number of RAW blocks to record.

    Raises:
        ValueError: If the request is underspecified and no fallback is available.
    """
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


def _resolve_total_obs_num_samples(
    length_spec: _RecordLengthSpec,
    *,
    num_antennas: int = 1,
    sample_rate: float = 3e9,
    block_size: int = 134217728,
    num_bits: int = 8,
    num_pols: int = 2,
    num_branches: int = 1024,
    num_chans: int = 64,
) -> int:
    """Estimate total pre-channelization time samples for an observation request.

    Args:
        length_spec: Normalized record-length specification.
        num_antennas: Number of antennas.
        sample_rate: Time-domain sample rate in Hz.
        block_size: RAW block size in bytes.
        num_bits: Requantized bit depth.
        num_pols: Number of recorded polarizations.
        num_branches: Number of PFB branches.
        num_chans: Number of recorded coarse channels.

    Returns:
        Estimated total count of real voltage samples.
    """
    tbin = num_branches / sample_rate
    chan_bw = 1 / tbin
    bytes_per_sample = 2 * num_pols * num_bits / 8
    num_blocks = _resolve_num_blocks(
        length_spec,
        get_num_blocks=lambda obs_length: int(
            obs_length
            * chan_bw
            * num_antennas
            * num_chans
            * bytes_per_sample
            / block_size
        ),
    )
    samples_per_block = int(block_size / (num_antennas * num_chans * bytes_per_sample))
    return num_blocks * samples_per_block * num_branches
