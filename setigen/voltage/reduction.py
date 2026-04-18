from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Literal

import numpy as np
from astropy import units as u

from ..frame import Frame
from ._reduction import (
    _build_reduction_metadata,
    _channelize_block,
    _create_writer,
    _decode_raw_block,
    _iter_raw_data_blocks,
    _resolve_raw_input,
)
from ._reduction.writers import _build_filterbank_header


class PolarizationMode(IntEnum):
    TOTAL_POWER = 1
    FULL_POL = 4
    FULL_STOKES = -4


@dataclass(frozen=True)
class RawReductionSpec:
    fftlength: int
    integration_factor: int
    pol_mode: int
    output_format: Literal["fil", "h5"]
    start_chan: int | None = None
    num_chans: int | None = None
    backend: Literal["auto", "numpy", "cupy"] = "auto"

    def __post_init__(self):
        if self.fftlength <= 0:
            raise ValueError("fftlength must be positive.")
        if self.integration_factor <= 0:
            raise ValueError("integration_factor must be positive.")
        if self.pol_mode not in tuple(mode.value for mode in PolarizationMode):
            raise ValueError(f"Unsupported polarization mode '{self.pol_mode}'.")
        if self.output_format not in ("fil", "h5"):
            raise ValueError(f"Unsupported output format '{self.output_format}'.")
        if self.backend not in ("auto", "numpy", "cupy"):
            raise ValueError(f"Unsupported reduction backend '{self.backend}'.")
        if self.start_chan is not None and self.start_chan < 0:
            raise ValueError("start_chan must be non-negative.")
        if self.num_chans is not None and self.num_chans <= 0:
            raise ValueError("num_chans must be positive.")


def _reduce_chunks(input_path, spec, *, max_blocks=None):
    input_spec = _resolve_raw_input(input_path)
    metadata = _build_reduction_metadata(input_spec, spec)

    for data_chunk in _iter_raw_data_blocks(input_spec, max_blocks=max_blocks):
        voltages = _decode_raw_block(data_chunk,
                                     num_bits=input_spec.num_bits,
                                     num_pols=input_spec.num_pols,
                                     num_chans=input_spec.num_chans,
                                     start_chan=metadata.start_chan,
                                     num_selected_chans=metadata.num_chans)
        reduced = _channelize_block(voltages,
                                    fftlength=spec.fftlength,
                                    integration_factor=spec.integration_factor,
                                    pol_mode=spec.pol_mode,
                                    backend=spec.backend)
        if reduced is not None and reduced.shape[0] > 0:
            yield reduced, input_spec, metadata


def reduce_raw(input_path, output_path, spec, *, overwrite=False, tmp_dir=None):
    output_path = Path(output_path)

    chunk_iter = _reduce_chunks(input_path, spec)
    first = next(chunk_iter, None)
    if first is None:
        raise ValueError("No spectra were produced from the supplied RAW input and reduction settings.")

    first_chunk, input_spec, metadata = first
    header = _build_filterbank_header(input_spec, metadata, pol_mode=spec.pol_mode)

    with _create_writer(output_path,
                        output_format=spec.output_format,
                        overwrite=overwrite,
                        tmp_dir=tmp_dir,
                        header=header,
                        total_fchans=metadata.total_fchans,
                        nifs=metadata.nifs) as writer:
        writer.append(first_chunk)
        for chunk, _, _ in chunk_iter:
            writer.append(chunk)

    return output_path


def reduce_raw_to_frame(input_path, spec, *, max_blocks=None):
    if spec.pol_mode != PolarizationMode.TOTAL_POWER:
        raise ValueError("reduce_raw_to_frame only supports total-power / Stokes-I reduction.")

    chunks = []
    input_spec = None
    metadata = None
    for chunk, input_spec, metadata in _reduce_chunks(input_path, spec, max_blocks=max_blocks):
        chunks.append(chunk[:, 0, :])

    if not chunks or input_spec is None or metadata is None:
        raise ValueError("No frame data were produced from the supplied RAW input and reduction settings.")

    data = np.concatenate(chunks, axis=0)
    return Frame.from_data(df=metadata.df_hz * u.Hz,
                           dt=metadata.dt_s * u.s,
                           fch1=metadata.fch1_hz * u.Hz,
                           ascending=metadata.ascending,
                           data=data,
                           metadata={
                               "raw_input_stem": str(input_spec.stem),
                               "pol_mode": int(spec.pol_mode),
                               "fftlength": spec.fftlength,
                               "integration_factor": spec.integration_factor,
                           })
