from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np


_DEFAULT_MAX_WORKING_SET_BYTES = 512 * 1024**2


@dataclass(frozen=True)
class _SubblockPlan:
    """Computed partitioning for one RAW block."""

    total_time_samples: int
    num_subblocks: int
    samples_per_subblock: int
    bytes_per_subblock: int
    initial_windows: int


def _estimate_working_set_bytes(backend: Any, *, coarse_time_samples: int) -> int:
    """Estimate transient working-set size for one subblock.

    Args:
        backend: Raw-voltage backend with filterbank and recording geometry.
        coarse_time_samples: Number of coarse time samples to process in one
            subblock.

    Returns:
        Estimated transient working-set size in bytes.
    """
    raw_time_samples = (coarse_time_samples + backend.num_taps) * backend.num_branches
    raw_bytes = (
        backend.num_antennas
        * backend.num_pols
        * raw_time_samples
        * np.dtype(np.float64).itemsize
    )
    pfb_frontend_bytes = (
        coarse_time_samples
        * backend.num_branches
        * np.dtype(np.float64).itemsize
    )
    fft_bytes = (
        coarse_time_samples
        * backend.num_branches
        * np.dtype(np.complex128).itemsize
        // 2
    )
    return int(raw_bytes + pfb_frontend_bytes + fft_bytes)


def _resolve_subblock_budget(backend: Any, *, total_time_samples: int) -> int:
    """Choose a safe minimum subblock count for the configured memory budget.

    Args:
        backend: Raw-voltage backend with memory-budget settings.
        total_time_samples: Total coarse time samples in one output RAW block.

    Returns:
        Minimum number of subblocks required to respect the configured
        working-set budget.
    """
    max_working_set_bytes = getattr(
        backend,
        "max_working_set_bytes",
        _DEFAULT_MAX_WORKING_SET_BYTES,
    )
    if max_working_set_bytes is None or max_working_set_bytes <= 0:
        return max(1, int(backend.num_subblocks))

    max_coarse_samples = backend.num_taps
    while (
        _estimate_working_set_bytes(
            backend,
            coarse_time_samples=max_coarse_samples,
        )
        <= max_working_set_bytes
    ):
        candidate = max_coarse_samples + backend.num_taps
        if candidate > total_time_samples:
            max_coarse_samples = total_time_samples
            break
        max_coarse_samples = candidate

    budgeted_subblocks = int(np.ceil(total_time_samples / max_coarse_samples))
    return max(1, int(backend.num_subblocks), budgeted_subblocks)


def _plan_subblocks(backend: Any, *, obsnchan: int) -> _SubblockPlan:
    """Plan subblock sizing for one output RAW block.

    Args:
        backend: Raw-voltage backend with recording geometry.
        obsnchan: Number of output coarse-channel streams in the RAW block.

    Returns:
        Planned subblock layout for one RAW block.

    Raises:
        ValueError: If the block size is incompatible with the subblock stride.
    """
    stride = int(obsnchan * backend.num_taps * backend.bytes_per_sample)
    if backend.block_size % stride != 0:
        raise ValueError("block_size must be divisible by the backend subblock stride.")

    total_time_samples = int(backend.block_size / (obsnchan * backend.bytes_per_sample))
    requested_subblocks = _resolve_subblock_budget(
        backend,
        total_time_samples=total_time_samples,
    )
    initial_windows = int(np.ceil(total_time_samples / backend.num_taps / requested_subblocks)) + 1
    samples_per_subblock = backend.num_taps * (initial_windows - 1)
    num_subblocks = int(np.ceil(total_time_samples / samples_per_subblock))
    bytes_per_subblock = int(samples_per_subblock * backend.bytes_per_sample)

    return _SubblockPlan(
        total_time_samples=total_time_samples,
        num_subblocks=num_subblocks,
        samples_per_subblock=samples_per_subblock,
        bytes_per_subblock=bytes_per_subblock,
        initial_windows=initial_windows,
    )


def _get_windows_for_subblock(plan: _SubblockPlan, backend: Any, subblock: int) -> int:
    """Return the number of PFB windows required for a subblock.

    Args:
        plan: Planned subblock layout.
        backend: Raw-voltage backend.
        subblock: Zero-based subblock index.

    Returns:
        Number of PFB windows to request for this subblock.
    """
    windows = plan.initial_windows
    if (
        plan.total_time_samples % plan.samples_per_subblock != 0
        and subblock == plan.num_subblocks - 1
    ):
        windows = int(
            (plan.total_time_samples % plan.samples_per_subblock) / backend.num_taps
        ) + 1
    return windows


def _get_num_samples_for_subblock(backend: Any, *, windows: int) -> int:
    """Return the number of raw time-domain samples needed for a subblock.

    Args:
        backend: Raw-voltage backend.
        windows: Number of PFB windows requested for this subblock.

    Returns:
        Number of raw time-domain voltage samples to generate.
    """
    if backend.antenna_source.start_obs:
        return backend.num_branches * backend.num_taps * windows
    return backend.num_branches * backend.num_taps * (windows - 1)


def _get_subblock_time_range(
    plan: _SubblockPlan,
    backend: Any,
    *,
    subblock: int,
    windows: int,
) -> int:
    """Return the byte span occupied by one subblock in the packed RAW block.

    Args:
        plan: Planned subblock layout.
        backend: Raw-voltage backend.
        subblock: Zero-based subblock index.
        windows: Number of PFB windows requested for this subblock.

    Returns:
        Number of bytes written by this subblock into the packed RAW buffer.
    """
    if (
        plan.total_time_samples % plan.samples_per_subblock != 0
        and subblock == plan.num_subblocks - 1
    ):
        return backend.num_taps * (windows - 1) * backend.bytes_per_sample
    return plan.bytes_per_subblock


def _get_time_indices(
    backend: Any,
    *,
    subblock: int,
    bytes_per_subblock: int,
    subblock_time_range: int,
    pol: int,
) -> np.ndarray:
    """Return byte indices for one polarization inside the final RAW buffer.

    Args:
        backend: Raw-voltage backend.
        subblock: Zero-based subblock index.
        bytes_per_subblock: Packed byte span reserved per subblock.
        subblock_time_range: Actual byte span written by this subblock.
        pol: Polarization index.

    Returns:
        Byte offsets for the selected polarization inside the packed buffer.
    """
    return (
        subblock * bytes_per_subblock
        + backend.num_bits // 4 * pol
        + np.arange(
            0,
            subblock_time_range,
            backend.num_bits // 4 * backend.num_pols,
        )
    )


def _channelize_voltage(
    backend: Any,
    v: Any,
    *,
    antenna: int,
    pol: int,
    digitize: bool,
) -> Any:
    """Digitize, PFB-channelize, and coarse-channel select one voltage stream.

    Args:
        backend: Raw-voltage backend.
        v: Raw time-domain voltages for one antenna and polarization.
        antenna: Antenna index.
        pol: Polarization index.
        digitize: Whether to digitize before PFB channelization.

    Returns:
        Complex coarse-channel voltages for the selected channel slice.
    """
    if digitize:
        start = time.time()
        v = backend.digitizer[antenna][pol].quantize(v)
        backend.digitizer_stage_t += time.time() - start

    start = time.time()
    v = backend.filterbank[antenna][pol].channelize(v, cache=True)
    v = v[:, backend.start_chan : backend.start_chan + backend.num_chans]
    backend.filterbank_stage_t += time.time() - start
    return v


def _requantize_voltage(
    backend: Any,
    v: Any,
    *,
    antenna: int,
    pol: int,
    c_idx: np.ndarray,
    t_idx: np.ndarray,
    input_voltages: Any,
    digitize: bool,
    xp: Any,
) -> Any:
    """Requantize one coarse-channelized voltage stream.

    Args:
        backend: Raw-voltage backend.
        v: Complex coarse-channel voltages.
        antenna: Antenna index.
        pol: Polarization index.
        c_idx: Coarse-channel indices in the packed RAW buffer.
        t_idx: Time or byte indices in the packed RAW buffer.
        input_voltages: Optional decoded RAW background block to mix in.
        digitize: Whether the input stream was digitized before channelization.
        xp: Numerical array module, either NumPy or CuPy.

    Returns:
        Requantized complex voltages ready for RAW packing.

    Raises:
        ValueError: If the backend bit width is unsupported.
    """
    start = time.time()

    if backend.input_file_stem is not None:
        requantizer = backend.requantizer[antenna][pol]

        temp_mean_r = requantizer.quantizer_r.target_mean
        requantizer.quantizer_r.target_mean = 0
        temp_mean_i = requantizer.quantizer_i.target_mean
        requantizer.quantizer_i.target_mean = 0

        if backend.filterbank[antenna][pol].channelized_stds is None:
            backend.filterbank[antenna][pol].estimate_channelized_stds()
        custom_stds = backend.filterbank[antenna][pol].channelized_stds

        if digitize:
            custom_stds *= backend.digitizer[antenna][pol].target_std
        v = requantizer.quantize(v, custom_stds=custom_stds)

        requantizer.quantizer_r.target_mean = temp_mean_r
        requantizer.quantizer_i.target_mean = temp_mean_i

        if backend.num_bits == 8:
            input_v = input_voltages[c_idx[:, np.newaxis], (t_idx // 2)[np.newaxis, :]]
        elif backend.num_bits == 4:
            input_v = input_voltages[c_idx[:, np.newaxis], t_idx[np.newaxis, :]]
        else:
            raise ValueError(f"{backend.num_bits} bits not supported...")
        input_v = xp.array(input_v)
        v += input_v.T

    v = backend.requantizer[antenna][pol].quantize(v)
    backend.requantizer_stage_t += time.time() - start
    return v


def _split_complex_components(v: Any, *, xp: Any) -> tuple[np.ndarray, np.ndarray]:
    """Split a complex array into real and imaginary components on the host.

    Args:
        v: Complex-valued array to split.
        xp: Numerical array module, either NumPy or CuPy.

    Returns:
        Tuple of host-side real and imaginary arrays.
    """
    try:
        real = xp.asnumpy(xp.real(v).T)
        imag = xp.asnumpy(xp.imag(v).T)
    except AttributeError:
        real = xp.real(v).T
        imag = xp.imag(v).T
    return real, imag


def _store_complex_components(
    backend: Any,
    final_voltages: np.ndarray,
    *,
    c_idx: np.ndarray,
    t_idx: np.ndarray,
    real: np.ndarray,
    imag: np.ndarray,
    requantize: bool,
) -> None:
    """Write one polarization's real and imaginary components into the RAW buffer.

    Args:
        backend: Raw-voltage backend.
        final_voltages: Final packed RAW output buffer.
        c_idx: Coarse-channel indices in the packed buffer.
        t_idx: Time or byte indices in the packed buffer.
        real: Real components to store.
        imag: Imaginary components to store.
        requantize: Whether values are already in the requantized RAW format.

    Raises:
        ValueError: If the backend bit width is unsupported.
    """
    if backend.num_bits == 8 or not requantize:
        final_voltages[c_idx[:, np.newaxis], t_idx[np.newaxis, :]] = real
        final_voltages[c_idx[:, np.newaxis], (t_idx + 1)[np.newaxis, :]] = imag
        return

    if backend.num_bits == 4:
        imag[imag < 0] += 16
        final_voltages[c_idx[:, np.newaxis], t_idx[np.newaxis, :]] = real * 16 + imag
        return

    raise ValueError(f"{backend.num_bits} bits not supported...")
