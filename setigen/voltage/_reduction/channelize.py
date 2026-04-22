from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import Any

import numpy as np


def _get_array_module(backend: str) -> Any:
    """Resolve the numerical array backend for reduction.

    Args:
        backend: Requested backend name.

    Returns:
        Numerical array module, usually NumPy or CuPy.

    Raises:
        ValueError: If the backend value is unsupported.
    """
    if backend == "numpy":
        return np
    if backend == "cupy":
        cupy = importlib.import_module("cupy")
        return cupy
    if backend == "auto":
        try:
            cupy = importlib.import_module("cupy")
        except ImportError:
            return np
        return cupy
    raise ValueError(f"Unsupported reduction backend '{backend}'.")


def _fftshifted_spectra(voltages: np.ndarray, *, fftlength: int, xp: Any) -> list[Any] | None:
    """Fine-channelize one RAW block and return per-polarization spectra.

    Args:
        voltages: Decoded coarse-channel voltages with time, coarse-channel,
            and polarization axes.
        fftlength: Fine-channel FFT length.
        xp: Numerical array module, either NumPy or CuPy.

    Returns:
        Per-polarization FFT-shifted spectra, or `None` if no complete FFT row
        can be formed.
    """
    trimmed = voltages[: (voltages.shape[0] // fftlength) * fftlength]
    if trimmed.shape[0] == 0:
        return None

    spectra_by_pol = []
    for pol in range(trimmed.shape[2]):
        samples = xp.asarray(trimmed[:, :, pol]).T
        samples = samples.reshape((samples.shape[0], samples.shape[1] // fftlength, fftlength))
        fft_vals = xp.fft.fft(samples, fftlength, axis=2) / fftlength**0.5
        fft_vals = xp.fft.fftshift(fft_vals, axes=2)
        spectra_by_pol.append(fft_vals.transpose(1, 0, 2))
    return spectra_by_pol


def _selected_fftshifted_spectra(
    voltages: np.ndarray,
    *,
    fftlength: int,
    channel_indices: Sequence[int],
    xp: Any,
) -> list[Any] | None:
    """Fine-channelize selected flattened frequency bins from one RAW block.

    Args:
        voltages: Decoded coarse-channel voltages with time, coarse-channel,
            and polarization axes.
        fftlength: Fine-channel FFT length.
        channel_indices: Flattened fine-channel indices to return, ordered as
            in the full coarse/fine flattened product.
        xp: Numerical array module, either NumPy or CuPy.

    Returns:
        Per-polarization FFT-shifted spectra with shape ``(time, frequency)``,
        or ``None`` if no complete FFT row can be formed.
    """
    trimmed = voltages[: (voltages.shape[0] // fftlength) * fftlength]
    if trimmed.shape[0] == 0:
        return None

    channel_indices = np.asarray(channel_indices, dtype=int)
    if channel_indices.size == 0:
        return None
    max_index = voltages.shape[1] * fftlength
    if np.any(channel_indices < 0) or np.any(channel_indices >= max_index):
        raise ValueError("Selected fine-channel indices are out of bounds.")

    coarse_indices = channel_indices // fftlength
    shifted_fine_indices = channel_indices % fftlength
    fft_bins = (shifted_fine_indices + fftlength // 2) % fftlength
    unique_coarse = np.unique(coarse_indices)

    sample_idx = xp.arange(fftlength)
    spectra_by_pol = []
    for pol in range(trimmed.shape[2]):
        samples = xp.asarray(trimmed[:, :, pol]).T
        samples = samples.reshape((samples.shape[0], samples.shape[1] // fftlength, fftlength))
        out = xp.empty((samples.shape[1], channel_indices.size), dtype=complex)
        for coarse_chan in unique_coarse:
            out_cols = np.where(coarse_indices == coarse_chan)[0]
            bins = xp.asarray(fft_bins[out_cols])
            twiddle = xp.exp(-2j * xp.pi * sample_idx[:, xp.newaxis] * bins[xp.newaxis, :] / fftlength)
            out[:, out_cols] = samples[coarse_chan] @ twiddle / fftlength**0.5
        spectra_by_pol.append(out)
    return spectra_by_pol


def _integrate_products(products: Any, *, integration_factor: int, xp: Any) -> Any:
    """Integrate fine-channelized products over consecutive time rows.

    Args:
        products: Fine-channelized spectrogram products.
        integration_factor: Number of time rows to sum together.
        xp: Numerical array module, either NumPy or CuPy.

    Returns:
        Integrated products, or `None` if no complete integration group exists.
    """
    trimmed = products[: (products.shape[0] // integration_factor) * integration_factor]
    if trimmed.shape[0] == 0:
        return None
    trimmed = trimmed.reshape(
        (trimmed.shape[0] // integration_factor, integration_factor, *trimmed.shape[1:])
    )
    return trimmed.sum(axis=1)


def _flatten_channels(products: Any) -> Any:
    """Flatten coarse-channel and fine-channel axes into one frequency axis.

    Args:
        products: Array with separate coarse- and fine-frequency axes.

    Returns:
        Array with a single flattened frequency axis.
    """
    return products.reshape(products.shape[0], products.shape[1] * products.shape[2])


def _channelize_block(
    voltages: np.ndarray,
    *,
    fftlength: int,
    integration_factor: int,
    pol_mode: int,
    backend: str,
    channel_indices: Sequence[int] | None = None,
    fine_method: str = "auto",
) -> np.ndarray | None:
    """Reduce one decoded RAW block into total-power or polarization products.

    Args:
        voltages: Decoded coarse-channel voltages from one RAW block.
        fftlength: Fine-channel FFT length.
        integration_factor: Number of spectra to integrate in time.
        pol_mode: Polarization output mode.
        backend: Numerical array backend name.
        channel_indices: Optional flattened fine-channel indices to return.
        fine_method: Fine-channel transform method. ``"full"`` computes the
            full FFT before slicing, ``"selected"`` computes only requested
            DFT bins, and ``"auto"`` uses selected DFT for small requests.

    Returns:
        Reduced spectrogram chunk, or `None` if no complete output rows are
        produced.

    Raises:
        ValueError: If the requested polarization mode is unsupported or
            incompatible with the decoded input.
    """
    if fine_method not in ("auto", "full", "selected"):
        raise ValueError("fine_method must be one of 'auto', 'full', or 'selected'.")

    xp = _get_array_module(backend)
    if channel_indices is not None:
        channel_indices = np.asarray(channel_indices, dtype=int)
        if fine_method == "auto":
            selected_limit = min(64, max(1, fftlength // 8))
            fine_method = "selected" if len(channel_indices) <= selected_limit else "full"
    if channel_indices is not None and fine_method == "selected":
        spectra = _selected_fftshifted_spectra(
            voltages,
            fftlength=fftlength,
            channel_indices=channel_indices,
            xp=xp,
        )
        selected_output = True
    else:
        spectra = _fftshifted_spectra(voltages, fftlength=fftlength, xp=xp)
        selected_output = False
    if spectra is None:
        return None

    xx = xp.abs(spectra[0]) ** 2
    yy = xp.abs(spectra[1]) ** 2 if len(spectra) > 1 else None

    if pol_mode == 1:
        total = xx if yy is None else xx + yy
        if channel_indices is not None and not selected_output:
            total = _flatten_channels(total)[:, channel_indices]
        integrated = _integrate_products(
            total,
            integration_factor=integration_factor,
            xp=xp,
        )
        if integrated is None:
            return None
        if selected_output or channel_indices is not None:
            result = integrated[:, np.newaxis, :]
        else:
            result = _flatten_channels(integrated)[:, np.newaxis, :]
    else:
        if yy is None:
            raise ValueError(
                "Full-pol and full-Stokes reduction require a dual-polarization RAW input."
            )
        xy = spectra[0] * xp.conj(spectra[1])
        re_xy = xp.real(xy)
        im_xy = xp.imag(xy)

        if channel_indices is not None and not selected_output:
            xx = _flatten_channels(xx)[:, channel_indices]
            yy = _flatten_channels(yy)[:, channel_indices]
            re_xy = _flatten_channels(re_xy)[:, channel_indices]
            im_xy = _flatten_channels(im_xy)[:, channel_indices]

        if pol_mode == 4:
            products = xp.stack((xx, yy, re_xy, im_xy), axis=1)
        elif pol_mode == -4:
            products = xp.stack((xx + yy, xx - yy, 2 * re_xy, -2 * im_xy), axis=1)
        else:
            raise ValueError(f"Unsupported polarization mode '{pol_mode}'.")

        integrated = _integrate_products(
            products,
            integration_factor=integration_factor,
            xp=xp,
        )
        if integrated is None:
            return None
        if selected_output or channel_indices is not None:
            result = integrated
        else:
            result = integrated.reshape(
                integrated.shape[0],
                integrated.shape[1],
                integrated.shape[2] * integrated.shape[3],
            )

    if xp is not np:
        result = xp.asnumpy(result)
    return np.asarray(result, dtype=np.float32)
