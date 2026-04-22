from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class _ReductionMetadata:
    """Metadata for one reduced spectrogram product."""

    start_chan: int
    num_chans: int
    nifs: int
    df_hz: float
    dt_s: float
    fch1_hz: float
    ascending: bool
    total_fchans: int
    channel_start: int = 0
    channel_stop: int | None = None


def _validate_selection_args(reduction_spec: Any) -> None:
    """Validate mutually exclusive coarse and frequency selections.

    Args:
        reduction_spec: Reduction configuration.

    Raises:
        ValueError: If mutually exclusive selections are supplied together.
    """
    has_coarse = reduction_spec.start_chan is not None or reduction_spec.num_chans is not None
    has_frequency = getattr(reduction_spec, "frequency_range", None) is not None
    if has_coarse and has_frequency:
        raise ValueError("frequency_range is mutually exclusive with start_chan/num_chans.")


def _frequency_channel_slice(
    *,
    fch1_hz: float,
    df_hz: float,
    total_fchans: int,
    ascending: bool,
    frequency_range: tuple[float, float],
) -> tuple[int, int, float]:
    """Resolve a frequency range to a contiguous flattened channel slice.

    Args:
        fch1_hz: Frequency of the first flattened channel in Hz.
        df_hz: Positive channel spacing in Hz.
        total_fchans: Total number of flattened fine channels before slicing.
        ascending: Whether frequencies increase with channel index.
        frequency_range: Inclusive frequency bounds in Hz.

    Returns:
        Tuple of start index, stop index, and selected first-channel frequency.

    Raises:
        ValueError: If no channels intersect the requested range.
    """
    f_min, f_max = sorted(float(f) for f in frequency_range)
    sign = 1 if ascending else -1
    freqs = fch1_hz + sign * df_hz * np.arange(total_fchans)
    selected = np.flatnonzero((freqs >= f_min) & (freqs <= f_max))
    if selected.size == 0:
        raise ValueError("frequency_range does not overlap the reduced frequency axis.")
    start = int(selected[0])
    stop = int(selected[-1] + 1)
    return start, stop, float(freqs[start])


def _build_reduction_metadata(input_spec: Any, reduction_spec: Any) -> _ReductionMetadata:
    """Derive output frequency and time metadata from RAW input and reduction spec.

    Args:
        input_spec: Normalized RAW input description.
        reduction_spec: Normalized reduction configuration.

    Returns:
        Derived metadata for the reduced output product.

    Raises:
        ValueError: If the requested coarse-channel slice is out of bounds.
    """
    _validate_selection_args(reduction_spec)

    start_chan = 0 if reduction_spec.start_chan is None else reduction_spec.start_chan
    num_chans = (
        input_spec.num_chans - start_chan
        if reduction_spec.num_chans is None
        else reduction_spec.num_chans
    )

    if start_chan < 0 or num_chans <= 0 or start_chan + num_chans > input_spec.num_chans:
        raise ValueError("Requested coarse channel slice is out of bounds for the RAW input.")

    nifs = 1 if reduction_spec.pol_mode == 1 else 4
    df_hz = abs(input_spec.chan_bw) / reduction_spec.fftlength
    dt_s = input_spec.tbin * reduction_spec.fftlength * reduction_spec.integration_factor
    coarse_fch1 = input_spec.fch1 + start_chan * input_spec.chan_bw
    fch1_hz = coarse_fch1 - input_spec.chan_bw / 2
    total_fchans = num_chans * reduction_spec.fftlength
    channel_start = 0
    channel_stop = total_fchans

    frequency_range = getattr(reduction_spec, "frequency_range", None)
    if frequency_range is not None:
        channel_start, channel_stop, fch1_hz = _frequency_channel_slice(
            fch1_hz=fch1_hz,
            df_hz=df_hz,
            total_fchans=total_fchans,
            ascending=input_spec.ascending,
            frequency_range=frequency_range,
        )
        total_fchans = channel_stop - channel_start

    return _ReductionMetadata(
        start_chan=start_chan,
        num_chans=num_chans,
        nifs=nifs,
        df_hz=df_hz,
        dt_s=dt_s,
        fch1_hz=fch1_hz,
        ascending=input_spec.ascending,
        total_fchans=total_fchans,
        channel_start=channel_start,
        channel_stop=channel_stop,
    )
