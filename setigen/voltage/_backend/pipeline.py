from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np


@dataclass(frozen=True)
class _SubblockPlan:
    total_time_samples: int
    num_subblocks: int
    samples_per_subblock: int
    bytes_per_subblock: int
    initial_windows: int


def _plan_subblocks(backend, *, obsnchan):
    if backend.block_size % int(obsnchan * backend.num_taps * backend.bytes_per_sample) != 0:
        raise ValueError("block_size must be divisible by the backend subblock stride.")

    total_time_samples = int(backend.block_size / (obsnchan * backend.bytes_per_sample))
    initial_windows = int(np.ceil(total_time_samples / backend.num_taps / backend.num_subblocks)) + 1
    samples_per_subblock = backend.num_taps * (initial_windows - 1)
    num_subblocks = int(np.ceil(total_time_samples / samples_per_subblock))
    bytes_per_subblock = int(samples_per_subblock * backend.bytes_per_sample)

    return _SubblockPlan(total_time_samples=total_time_samples,
                         num_subblocks=num_subblocks,
                         samples_per_subblock=samples_per_subblock,
                         bytes_per_subblock=bytes_per_subblock,
                         initial_windows=initial_windows)


def _get_windows_for_subblock(plan, backend, subblock):
    windows = plan.initial_windows
    if (plan.total_time_samples % plan.samples_per_subblock != 0
            and subblock == plan.num_subblocks - 1):
        windows = int((plan.total_time_samples % plan.samples_per_subblock) / backend.num_taps) + 1
    return windows


def _get_num_samples_for_subblock(backend, *, windows):
    if backend.antenna_source.start_obs:
        return backend.num_branches * backend.num_taps * windows
    return backend.num_branches * backend.num_taps * (windows - 1)


def _get_subblock_time_range(plan, backend, *, subblock, windows):
    if (plan.total_time_samples % plan.samples_per_subblock != 0
            and subblock == plan.num_subblocks - 1):
        return backend.num_taps * (windows - 1) * backend.bytes_per_sample
    return plan.bytes_per_subblock


def _get_time_indices(backend,
                      *,
                      subblock,
                      bytes_per_subblock,
                      subblock_time_range,
                      pol):
    return (subblock * bytes_per_subblock
            + backend.num_bits // 4 * pol
            + np.arange(0,
                        subblock_time_range,
                        backend.num_bits // 4 * backend.num_pols))


def _channelize_voltage(backend, v, *, antenna, pol, digitize):
    if digitize:
        start = time.time()
        v = backend.digitizer[antenna][pol].quantize(v)
        backend.digitizer_stage_t += time.time() - start

    start = time.time()
    v = backend.filterbank[antenna][pol].channelize(v, cache=True)
    v = v[:, backend.start_chan:backend.start_chan + backend.num_chans]
    backend.filterbank_stage_t += time.time() - start
    return v


def _requantize_voltage(backend,
                        v,
                        *,
                        antenna,
                        pol,
                        c_idx,
                        t_idx,
                        input_voltages,
                        digitize,
                        xp):
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


def _split_complex_components(v, *, xp):
    try:
        real = xp.asnumpy(xp.real(v).T)
        imag = xp.asnumpy(xp.imag(v).T)
    except AttributeError:
        real = xp.real(v).T
        imag = xp.imag(v).T
    return real, imag


def _store_complex_components(backend,
                              final_voltages,
                              *,
                              c_idx,
                              t_idx,
                              real,
                              imag,
                              requantize):
    if backend.num_bits == 8 or not requantize:
        final_voltages[c_idx[:, np.newaxis], t_idx[np.newaxis, :]] = real
        final_voltages[c_idx[:, np.newaxis], (t_idx + 1)[np.newaxis, :]] = imag
        return

    if backend.num_bits == 4:
        imag[imag < 0] += 16
        final_voltages[c_idx[:, np.newaxis], t_idx[np.newaxis, :]] = real * 16 + imag
        return

    raise ValueError(f"{backend.num_bits} bits not supported...")
