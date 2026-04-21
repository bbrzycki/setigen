from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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

    return _ReductionMetadata(
        start_chan=start_chan,
        num_chans=num_chans,
        nifs=nifs,
        df_hz=df_hz,
        dt_s=dt_s,
        fch1_hz=fch1_hz,
        ascending=input_spec.ascending,
        total_fchans=num_chans * reduction_spec.fftlength,
    )
