from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import numpy as np
from astropy import units as u

from ..frame import Frame
from ._backend.headers import _read_next_block
from ._backend.orchestration import _get_blocks_to_write, _get_num_output_files, _reset_recording_state
from ._backend.pipeline import (
    _channelize_voltage,
    _get_num_samples_for_subblock,
    _get_subblock_time_range,
    _get_time_indices,
    _get_windows_for_subblock,
    _plan_subblocks,
    _requantize_voltage,
)
from ._backend.recording import _RecordConfig, _resolve_num_blocks
from ._reduction.channelize import _channelize_block
from ._reduction.frame_context import _frame_context_kwargs
from ._reduction.metadata import _ReductionMetadata, _build_reduction_metadata
from ._reduction.writers import _build_filterbank_header, _create_writer
from .reduction import PolarizationMode


@dataclass(frozen=True)
class VoltageSpectrogramSpec:
    """Configuration for direct voltage-backend spectrogram generation.

    Args:
        fftlength: Fine-channel FFT length.
        integration_factor: Number of fine spectra to integrate in time.
        pol_mode: Polarization output mode.
        start_chan: First coarse channel relative to the backend recorded span.
        num_chans: Number of coarse channels to reduce.
        frequency_range: Optional inclusive final-frequency bounds in Hz.
        backend: Array backend for fine channelization.
        accuracy: Accuracy policy. Only ``"exact"`` is implemented.
        coarse_method: Coarse PFB transform method.
        fine_method: Fine FFT transform method.
    """

    fftlength: int
    integration_factor: int
    pol_mode: int = PolarizationMode.TOTAL_POWER
    start_chan: int | None = None
    num_chans: int | None = None
    frequency_range: tuple[float, float] | None = None
    backend: Literal["auto", "numpy", "cupy"] = "auto"
    accuracy: Literal["exact", "approx_zoom"] = "exact"
    coarse_method: Literal["auto", "full", "selected"] = "auto"
    fine_method: Literal["auto", "full", "selected"] = "auto"

    def __post_init__(self) -> None:
        """Validate spectrogram generation settings."""
        if self.fftlength <= 0:
            raise ValueError("fftlength must be positive.")
        if self.integration_factor <= 0:
            raise ValueError("integration_factor must be positive.")
        if self.pol_mode not in tuple(mode.value for mode in PolarizationMode):
            raise ValueError(f"Unsupported polarization mode '{self.pol_mode}'.")
        if self.backend not in ("auto", "numpy", "cupy"):
            raise ValueError(f"Unsupported backend '{self.backend}'.")
        if self.accuracy not in ("exact", "approx_zoom"):
            raise ValueError(f"Unsupported accuracy mode '{self.accuracy}'.")
        if self.accuracy != "exact":
            raise ValueError("Direct voltage spectrograms currently support only accuracy='exact'.")
        if self.coarse_method not in ("auto", "full", "selected"):
            raise ValueError(f"Unsupported coarse method '{self.coarse_method}'.")
        if self.fine_method not in ("auto", "full", "selected"):
            raise ValueError(f"Unsupported fine method '{self.fine_method}'.")
        if self.start_chan is not None and self.start_chan < 0:
            raise ValueError("start_chan must be non-negative.")
        if self.num_chans is not None and self.num_chans <= 0:
            raise ValueError("num_chans must be positive.")
        if self.frequency_range is not None:
            if self.start_chan is not None or self.num_chans is not None:
                raise ValueError("frequency_range is mutually exclusive with start_chan/num_chans.")
            if len(self.frequency_range) != 2:
                raise ValueError("frequency_range must contain exactly two frequency bounds.")


@dataclass(frozen=True)
class VoltageSpectrogramResult:
    """Spectrogram data and metadata produced by a voltage backend.

    Args:
        data: Spectrogram array with shape ``(time, nifs, frequency)``.
        metadata: Derived frequency/time metadata.
        spec: Generation specification.
        input_spec: Minimal input descriptor used for file headers.
    """

    data: np.ndarray
    metadata: _ReductionMetadata
    spec: VoltageSpectrogramSpec
    input_spec: Any

    def to_frame(self) -> Frame:
        """Return this total-power product as a :class:`setigen.Frame`.

        Returns:
            Total-power spectrogram as a :class:`setigen.Frame`.

        Raises:
            ValueError: If the result is not a total-power/Stokes-I product.
        """
        if self.spec.pol_mode != PolarizationMode.TOTAL_POWER:
            raise ValueError("to_frame only supports total-power / Stokes-I spectrograms.")
        return Frame.from_data(
            df=self.metadata.df_hz * u.Hz,
            dt=self.metadata.dt_s * u.s,
            fch1=self.metadata.fch1_hz * u.Hz,
            ascending=self.metadata.ascending,
            data=self.data[:, 0, :],
            metadata={
                "pol_mode": int(self.spec.pol_mode),
                "fftlength": self.spec.fftlength,
                "integration_factor": self.spec.integration_factor,
                "source": "voltage_backend",
            },
            **_frame_context_kwargs(self.input_spec,
                                    self.metadata,
                                    pol_mode=self.spec.pol_mode),
        )

    def write(
        self,
        output_path: str | Path,
        *,
        output_format: Literal["fil", "h5"] | None = None,
        overwrite: bool = False,
        tmp_dir: str | Path | None = None,
    ) -> Path:
        """Write the spectrogram as a filterbank product.

        Args:
            output_path: Destination path.
            output_format: Output format. If omitted, inferred from suffix.
            overwrite: Whether an existing output file may be replaced.
            tmp_dir: Optional staging directory.

        Returns:
            Final output path.
        """
        output_path = Path(output_path)
        if output_format is None:
            suffix = output_path.suffix.lower()
            if suffix == ".fil":
                output_format = "fil"
            elif suffix == ".h5":
                output_format = "h5"
            else:
                raise ValueError("output_format must be supplied when suffix is not .fil or .h5.")

        header = _build_filterbank_header(
            self.input_spec,
            self.metadata,
            pol_mode=self.spec.pol_mode,
        )
        with _create_writer(
            output_path,
            output_format=output_format,
            overwrite=overwrite,
            tmp_dir=tmp_dir,
            header=header,
            total_fchans=self.metadata.total_fchans,
            nifs=self.metadata.nifs,
        ) as writer:
            writer.append(self.data)
        return output_path


def _make_direct_input_spec(backend: Any) -> Any:
    """Build a minimal RAW-like input descriptor for direct products.

    Args:
        backend: Voltage backend providing observation geometry.

    Returns:
        Namespace containing the fields needed by reduction metadata and
        filterbank header builders.
    """
    return SimpleNamespace(
        num_chans=backend.num_chans,
        num_pols=backend.num_pols,
        num_bits=backend.num_bits,
        chan_bw=backend.chan_bw,
        tbin=backend.tbin,
        fch1=backend.fch1 + backend.start_chan * backend.chan_bw,
        ascending=backend.ascending,
        source_name="SYNTHETIC",
        rawdatafile="",
        tstart_mjd=0,
        stem="voltage_backend",
    )


def _to_host_array(array: Any, *, xp: Any) -> Any:
    """Return an array on the host, copying from CuPy when needed.

    Args:
        array: NumPy-like or CuPy-like array.
        xp: Numerical array module used to create ``array``.

    Returns:
        Host-side array when ``xp`` supports GPU arrays; otherwise ``array``.
    """
    try:
        return xp.asnumpy(array)
    except AttributeError:
        return array


def _collect_coarse_voltage_block(
    backend: Any,
    *,
    metadata: _ReductionMetadata,
    spec: VoltageSpectrogramSpec,
    digitize: bool,
    requantize: bool,
    verbose: bool,
    xp: Any,
) -> np.ndarray:
    """Collect one time-contiguous coarse-voltage block without RAW packing.

    Args:
        backend: Voltage backend to sample and channelize.
        metadata: Derived spectrogram metadata.
        spec: Spectrogram generation specification.
        digitize: Whether to quantize input voltages before coarse PFB.
        requantize: Whether to quantize complex post-PFB voltages.
        verbose: Whether progress output is enabled.
        xp: Numerical array module used by the backend.

    Returns:
        Complex coarse-voltage block with time, coarse-channel, and
        polarization axes.
    """
    original_obsnchan = backend.num_chans * backend.num_antennas
    selected_num_chans = metadata.num_chans
    selected_obsnchan = selected_num_chans * backend.num_antennas
    plan = _plan_subblocks(backend, obsnchan=original_obsnchan)
    backend.num_subblocks = plan.num_subblocks
    block_voltages = np.empty(
        (plan.total_time_samples, selected_obsnchan, backend.num_pols),
        dtype=np.complex64,
    )

    if backend.input_file_stem is not None:
        if not requantize:
            raise ValueError("Must set 'requantize=True' when using input RAW data.")
        input_voltages = _read_next_block(backend)
    else:
        input_voltages = None

    absolute_start_chan = backend.start_chan + metadata.start_chan
    relative_start_chan = metadata.start_chan
    row_offset = 0

    for subblock in range(plan.num_subblocks):
        windows = _get_windows_for_subblock(plan, backend, subblock)
        num_samples = _get_num_samples_for_subblock(backend, windows=windows)
        antennas_v = backend.antenna_source.get_samples(num_samples)

        subblock_t_range = _get_subblock_time_range(
            plan,
            backend,
            subblock=subblock,
            windows=windows,
        )
        rows_written = None
        for antenna in range(backend.num_antennas):
            local_c_idx = antenna * selected_num_chans + np.arange(selected_num_chans)
            input_c_idx = (
                antenna * backend.num_chans
                + relative_start_chan
                + np.arange(selected_num_chans)
            )
            for pol in range(backend.num_pols):
                t_idx = _get_time_indices(
                    backend,
                    subblock=subblock,
                    bytes_per_subblock=plan.bytes_per_subblock,
                    subblock_time_range=subblock_t_range,
                    pol=pol,
                )
                v = _channelize_voltage(
                    backend,
                    antennas_v[antenna][pol],
                    antenna=antenna,
                    pol=pol,
                    digitize=digitize,
                    start_chan=absolute_start_chan,
                    num_chans=selected_num_chans,
                    coarse_method=spec.coarse_method,
                )
                if requantize:
                    v = _requantize_voltage(
                        backend,
                        v,
                        antenna=antenna,
                        pol=pol,
                        c_idx=input_c_idx,
                        t_idx=t_idx,
                        input_voltages=input_voltages,
                        digitize=digitize,
                        xp=xp,
                    )
                if rows_written is None:
                    rows_written = v.shape[0]
                block_voltages[
                    row_offset:row_offset + v.shape[0],
                    local_c_idx,
                    pol,
                ] = np.asarray(_to_host_array(v, xp=xp), dtype=np.complex64)
        row_offset += 0 if rows_written is None else rows_written

    return block_voltages[:row_offset]


def generate_voltage_spectrogram(
    backend: Any,
    spec: VoltageSpectrogramSpec,
    *,
    obs_length: float | None = None,
    num_blocks: int | None = None,
    length_mode: str = "obs_length",
    digitize: bool = True,
    requantize: bool = True,
    verbose: bool = True,
    xp: Any,
) -> VoltageSpectrogramResult:
    """Generate a spectrogram directly from a voltage backend.

    Args:
        backend: Configured :class:`RawVoltageBackend`.
        spec: Spectrogram generation specification.
        obs_length: Observation length in seconds.
        num_blocks: Number of backend blocks to generate.
        length_mode: Strategy for interpreting observation length inputs.
        digitize: Whether to quantize input voltages before the PFB.
        requantize: Whether to quantize complex post-PFB voltages.
        verbose: Whether to emit progress output.
        xp: Numerical array module used by the backend.

    Returns:
        Direct spectrogram result.
    """
    input_spec = _make_direct_input_spec(backend)
    metadata = _build_reduction_metadata(input_spec, spec)
    record_config = _RecordConfig.from_values(
        obs_length=obs_length,
        num_blocks=num_blocks,
        length_mode=length_mode,
        header_dict=None,
        digitize=digitize,
        load_template=False,
        verbose=verbose,
    )
    backend.num_blocks = _resolve_num_blocks(
        record_config.length,
        get_num_blocks=backend.get_num_blocks,
        fallback_num_blocks=backend.input_num_blocks,
    )
    if backend.input_num_blocks is not None:
        backend.num_blocks = min(backend.num_blocks, backend.input_num_blocks)
    backend.obs_length = backend.num_blocks * backend.time_per_block
    backend.total_obs_num_samples = int(backend.obs_length / backend.tbin) * backend.num_branches

    _reset_recording_state(backend)

    channel_indices = None
    full_fchans = metadata.num_chans * spec.fftlength
    if metadata.channel_start != 0 or metadata.channel_stop != full_fchans:
        channel_indices = np.arange(metadata.channel_start, metadata.channel_stop)

    chunks: list[np.ndarray] = []
    num_files = _get_num_output_files(backend, xp=xp)
    for file_index in range(num_files):
        blocks_to_write = _get_blocks_to_write(
            backend,
            file_index=file_index,
            num_files=num_files,
        )
        input_context = nullcontext(None)
        if backend.input_file_stem is not None:
            input_context = open(f"{backend.input_file_stem}.{file_index:04}.raw", "rb")
        with input_context as input_file_handler:
            backend.input_file_handler = input_file_handler
            for _ in range(blocks_to_write):
                coarse_voltages = _collect_coarse_voltage_block(
                    backend,
                    metadata=metadata,
                    spec=spec,
                    digitize=digitize,
                    requantize=requantize,
                    verbose=verbose,
                    xp=xp,
                )
                reduced = _channelize_block(
                    coarse_voltages,
                    fftlength=spec.fftlength,
                    integration_factor=spec.integration_factor,
                    pol_mode=spec.pol_mode,
                    backend=spec.backend,
                    channel_indices=channel_indices,
                    fine_method=spec.fine_method,
                )
                if reduced is not None and reduced.shape[0] > 0:
                    chunks.append(reduced)
            backend.input_file_handler = None

    if not chunks:
        raise ValueError("No spectra were produced from the supplied voltage settings.")
    return VoltageSpectrogramResult(
        data=np.concatenate(chunks, axis=0),
        metadata=metadata,
        spec=spec,
        input_spec=input_spec,
    )
