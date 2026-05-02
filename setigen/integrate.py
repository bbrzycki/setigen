from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np
from astropy.stats import sigma_clip
from . import utils
from ._frame.context import _finalize_derived_frame, _source_bounds_metadata
from .spectrum import Spectrum 
from .timeseries import TimeSeries


class IntegrationAxis(str, Enum):
    """Supported axes for frame integration."""

    TIME = "t"
    FREQUENCY = "f"


class IntegrationMode(str, Enum):
    """Supported frame-integration modes."""

    MEAN = "mean"
    SUM = "sum"


def _resolve_integration_axis(axis: IntegrationAxis | str | int) -> IntegrationAxis:
    """Normalize a user-supplied integration axis.

    Args:
        axis: Raw axis selector.

    Returns:
        Normalized integration-axis enum value.
    """
    if isinstance(axis, IntegrationAxis):
        return axis
    if isinstance(axis, str):
        normalized = axis.lower()
        if normalized in {"f", "freq", "frequency", "frequencies"}:
            return IntegrationAxis.FREQUENCY
        return IntegrationAxis.TIME
    if axis == 1:
        return IntegrationAxis.FREQUENCY
    return IntegrationAxis.TIME


def _resolve_integration_mode(mode: IntegrationMode | str) -> IntegrationMode:
    """Normalize a user-supplied integration mode.

    Args:
        mode: Raw integration-mode selector.

    Returns:
        Normalized integration-mode enum value.
    """
    if isinstance(mode, IntegrationMode):
        return mode
    if isinstance(mode, str) and mode[:1].lower() == IntegrationMode.SUM.value[:1]:
        return IntegrationMode.SUM
    return IntegrationMode.MEAN


def _resolve_region(
    fr: Any,
    *,
    f_range: tuple[Any, Any] | None = None,
    t_range: tuple[Any, Any] | None = None,
    f_index_range: tuple[int, int] | None = None,
    t_index_range: tuple[int, int] | None = None,
) -> tuple[int, int, int, int]:
    """Resolve optional frame-region selectors.

    Args:
        fr: Frame-like input or array.
        f_range: Optional physical frequency range.
        t_range: Optional relative time range.
        f_index_range: Optional half-open frequency index range.
        t_index_range: Optional half-open time index range.

    Returns:
        Half-open frequency and time bounds.
    """
    range_values = (f_range, t_range, f_index_range, t_index_range)
    if not hasattr(fr, "_resolve_frequency_index_range"):
        if any(value is not None for value in range_values):
            raise ValueError("Region selections require a Frame-like input")
        data = np.asarray(utils.array(fr))
        return 0, data.shape[1], 0, data.shape[0]

    f_start, f_stop = fr._resolve_frequency_index_range(
        f_range=f_range,
        f_index_range=f_index_range,
    )
    t_start, t_stop = fr._resolve_time_index_range(
        t_range=t_range,
        t_index_range=t_index_range,
    )
    if f_start >= f_stop or t_start >= t_stop:
        raise ValueError("Requested integration region is empty")
    return f_start, f_stop, t_start, t_stop


def _chunk_limits(
    fr: Any,
    *,
    fchans: int,
    max_chunk_bytes: int | None,
) -> tuple[int, int]:
    """Choose conservative file-backed time/frequency chunk limits.

    Args:
        fr: File-backed frame.
        fchans: Selected frequency-channel count.
        max_chunk_bytes: Optional byte budget.

    Returns:
        Tuple of `(time_chunk_tchans, frequency_chunk_fchans)`.
    """
    if max_chunk_bytes is None:
        max_chunk_bytes = getattr(fr, "_max_chunk_bytes", 256 * 1024 * 1024)
    if max_chunk_bytes < 1:
        raise ValueError("max_chunk_bytes must be positive")

    backend = fr._file_backend
    itemsize = np.dtype(getattr(backend, "dtype", np.float32)).itemsize
    f_chunk = max(1, min(fchans, max_chunk_bytes // itemsize))
    t_chunk = max(1, max_chunk_bytes // max(f_chunk * itemsize, 1))
    return int(t_chunk), int(f_chunk)


def _integrate_eager_region(
    fr: Any,
    *,
    axis: IntegrationAxis,
    mode: IntegrationMode,
    f_start: int,
    f_stop: int,
    t_start: int,
    t_stop: int,
) -> np.ndarray:
    """Integrate an eager frame or array region.

    Args:
        fr: Eager frame-like input or array.
        axis: Axis to collapse.
        mode: Reduction mode.
        f_start: Frequency start index.
        f_stop: Frequency stop index.
        t_start: Time start index.
        t_stop: Time stop index.

    Returns:
        Reduced two-dimensional singleton-axis array.
    """
    if hasattr(fr, "data"):
        data = np.asarray(fr.data[t_start:t_stop, f_start:f_stop])
    else:
        data = np.asarray(utils.array(fr))[t_start:t_stop, f_start:f_stop]
    np_axis = 1 if axis is IntegrationAxis.FREQUENCY else 0
    if mode is IntegrationMode.SUM:
        return np.sum(data, axis=np_axis, keepdims=True)
    return np.mean(data, axis=np_axis, keepdims=True)


def _integrate_file_backed_region(
    fr: Any,
    *,
    axis: IntegrationAxis,
    mode: IntegrationMode,
    f_start: int,
    f_stop: int,
    t_start: int,
    t_stop: int,
    max_chunk_bytes: int | None,
) -> np.ndarray:
    """Integrate a file-backed frame region in bounded chunks.

    Args:
        fr: File-backed frame.
        axis: Axis to collapse.
        mode: Reduction mode.
        f_start: Frequency start index.
        f_stop: Frequency stop index.
        t_start: Time start index.
        t_stop: Time stop index.
        max_chunk_bytes: Optional byte budget.

    Returns:
        Reduced two-dimensional singleton-axis array.
    """
    backend = fr._file_backend
    selected_tchans = t_stop - t_start
    selected_fchans = f_stop - f_start
    t_chunk, f_chunk = _chunk_limits(fr,
                                     fchans=selected_fchans,
                                     max_chunk_bytes=max_chunk_bytes)

    if axis is IntegrationAxis.TIME:
        output = np.zeros(selected_fchans, dtype=np.float64)
        for local_f_start in range(0, selected_fchans, f_chunk):
            local_f_stop = min(selected_fchans, local_f_start + f_chunk)
            accumulator = np.zeros(local_f_stop - local_f_start, dtype=np.float64)
            for chunk_start in range(t_start, t_stop, t_chunk):
                chunk_stop = min(t_stop, chunk_start + t_chunk)
                data = backend.read_region(
                    chunk_start,
                    chunk_stop,
                    f_start + local_f_start,
                    f_start + local_f_stop,
                )
                accumulator += np.sum(data, axis=0)
            if mode is IntegrationMode.MEAN:
                accumulator /= selected_tchans
            output[local_f_start:local_f_stop] = accumulator
        return output[np.newaxis, :]

    output = np.zeros(selected_tchans, dtype=np.float64)
    for chunk_start in range(t_start, t_stop, t_chunk):
        chunk_stop = min(t_stop, chunk_start + t_chunk)
        row_accumulator = np.zeros(chunk_stop - chunk_start, dtype=np.float64)
        for local_f_start in range(0, selected_fchans, f_chunk):
            local_f_stop = min(selected_fchans, local_f_start + f_chunk)
            data = backend.read_region(
                chunk_start,
                chunk_stop,
                f_start + local_f_start,
                f_start + local_f_stop,
            )
            row_accumulator += np.sum(data, axis=1)
        if mode is IntegrationMode.MEAN:
            row_accumulator /= selected_fchans
        output[chunk_start - t_start:chunk_stop - t_start] = row_accumulator
    return output[:, np.newaxis]


def _normalize_reduced_data(data: np.ndarray) -> np.ndarray:
    """Sigma-normalize a reduced product.

    Args:
        data: Reduced singleton-axis product.

    Returns:
        Sigma-normalized data.
    """
    c_data = sigma_clip(data)
    return (data - np.mean(c_data)) / np.std(c_data)


def _region_fch1(fr: Any, f_start: int, f_stop: int) -> float:
    """Return frame-convention `fch1` for a selected frequency region.

    Args:
        fr: Source frame.
        f_start: Half-open frequency start index.
        f_stop: Half-open frequency stop index.

    Returns:
        First-channel frequency in the frame's storage convention.
    """
    if fr.ascending:
        return fr.fs[f_start]
    return fr.fs[f_stop - 1]


def _region_center_frequency(fr: Any, f_start: int, f_stop: int) -> float:
    """Return center frequency for a selected frequency region.

    Args:
        fr: Source frame.
        f_start: Half-open frequency start index.
        f_stop: Half-open frequency stop index.

    Returns:
        Center frequency in Hz.
    """
    return float((fr.frequency_edges[f_start] + fr.frequency_edges[f_stop]) / 2)


def integrate(
    fr: Any,
    axis: IntegrationAxis | str | int = 't',
    mode: IntegrationMode | str = 'mean',
    normalize: bool = False,
    as_frame: bool = False,
    *,
    f_range: tuple[Any, Any] | None = None,
    t_range: tuple[Any, Any] | None = None,
    f_index_range: tuple[int, int] | None = None,
    t_index_range: tuple[int, int] | None = None,
    max_chunk_bytes: int | None = None,
) -> np.ndarray | Spectrum | TimeSeries:
    """Integrate frame data over time or frequency.

    Args:
        fr: Input frame or two-dimensional array.
        axis: Axis over which to integrate.
        mode: Integration mode.
        normalize: Whether to sigma-normalize the integrated result.
        as_frame: Whether to return a `Spectrum` or `TimeSeries` object.
        f_range: Optional physical frequency range to select before reduction.
        t_range: Optional relative time range to select before reduction.
        f_index_range: Optional half-open frequency index range.
        t_index_range: Optional half-open time index range.
        max_chunk_bytes: Optional file-backed reduction memory budget.

    Returns:
        Integrated one-dimensional data or frame-like object.
    """
    resolved_axis = _resolve_integration_axis(axis)
    resolved_mode = _resolve_integration_mode(mode)
    f_start, f_stop, t_start, t_stop = _resolve_region(
        fr,
        f_range=f_range,
        t_range=t_range,
        f_index_range=f_index_range,
        t_index_range=t_index_range,
    )

    if getattr(fr, "is_file_backed", False):
        data = _integrate_file_backed_region(
            fr,
            axis=resolved_axis,
            mode=resolved_mode,
            f_start=f_start,
            f_stop=f_stop,
            t_start=t_start,
            t_stop=t_stop,
            max_chunk_bytes=max_chunk_bytes,
        )
    else:
        data = _integrate_eager_region(
            fr,
            axis=resolved_axis,
            mode=resolved_mode,
            f_start=f_start,
            f_stop=f_stop,
            t_start=t_start,
            t_stop=t_stop,
        )
        
    if normalize:
        data = _normalize_reduced_data(data)

    if as_frame:
        if not hasattr(fr, "df"):
            raise TypeError("as_frame=True requires a Frame-like input")
        source_bounds = _source_bounds_metadata(
            fr,
            f_index_range=(f_start, f_stop),
            t_index_range=(t_start, t_stop),
        )
        normalization = "sigma_clip" if normalize else None
        if resolved_axis is IntegrationAxis.FREQUENCY:
            # Time series
            new_fr = TimeSeries(df=fr.df * (f_stop - f_start),
                                dt=fr.dt,
                                fch1=_region_center_frequency(fr, f_start, f_stop),
                                ascending=fr.ascending,
                                data=data,
                                seed=fr.rng,
                                t_start=fr.t_start + t_start * fr.dt,
                                source_name=fr.source_name)
            product_type = "timeseries"
            collapsed_axis = "frequency"
        else:
            # Spectrum
            new_fr = Spectrum(df=fr.df,
                              dt=fr.dt * (t_stop - t_start),
                              fch1=_region_fch1(fr, f_start, f_stop),
                              ascending=fr.ascending,
                              data=data,
                              seed=fr.rng,
                              t_start=fr.t_start + t_start * fr.dt,
                              source_name=fr.source_name)
            product_type = "spectrum"
            collapsed_axis = "time"
        _finalize_derived_frame(
            fr,
            new_fr,
            operation="integrate",
            product_type=product_type,
            collapsed_axis=collapsed_axis,
            reducer=resolved_mode.value,
            normalization=normalization,
            source_bounds=source_bounds,
        )
        return new_fr
    else:
        return data.flatten()


def spectrum(
    fr: Any,
    mode: IntegrationMode | str = "mean",
    normalize: bool = False,
    *,
    f_range: tuple[Any, Any] | None = None,
    t_range: tuple[Any, Any] | None = None,
    f_index_range: tuple[int, int] | None = None,
    t_index_range: tuple[int, int] | None = None,
    max_chunk_bytes: int | None = None,
) -> Spectrum:
    """Produce a `Spectrum` from a spectrogram frame.

    Args:
        fr: Input frame or array-like object.
        mode: Integration mode.
        normalize: Whether to sigma-normalize the result.
        f_range: Optional physical frequency range to select before reduction.
        t_range: Optional relative time range to select before reduction.
        f_index_range: Optional half-open frequency index range.
        t_index_range: Optional half-open time index range.
        max_chunk_bytes: Optional file-backed reduction memory budget.

    Returns:
        Integrated spectrum.
    """
    return integrate(fr,
                     axis=IntegrationAxis.TIME,
                     mode=mode,
                     normalize=normalize,
                     as_frame=True,
                     f_range=f_range,
                     t_range=t_range,
                     f_index_range=f_index_range,
                     t_index_range=t_index_range,
                     max_chunk_bytes=max_chunk_bytes)


def timeseries(
    fr: Any,
    mode: IntegrationMode | str = "mean",
    normalize: bool = False,
    *,
    f_range: tuple[Any, Any] | None = None,
    t_range: tuple[Any, Any] | None = None,
    f_index_range: tuple[int, int] | None = None,
    t_index_range: tuple[int, int] | None = None,
    max_chunk_bytes: int | None = None,
) -> TimeSeries:
    """Produce a `TimeSeries` from a spectrogram frame.

    Args:
        fr: Input frame or array-like object.
        mode: Integration mode.
        normalize: Whether to sigma-normalize the result.
        f_range: Optional physical frequency range to select before reduction.
        t_range: Optional relative time range to select before reduction.
        f_index_range: Optional half-open frequency index range.
        t_index_range: Optional half-open time index range.
        max_chunk_bytes: Optional file-backed reduction memory budget.

    Returns:
        Integrated time series.
    """
    return integrate(fr,
                     axis=IntegrationAxis.FREQUENCY,
                     mode=mode,
                     normalize=normalize,
                     as_frame=True,
                     f_range=f_range,
                     t_range=t_range,
                     f_index_range=f_index_range,
                     t_index_range=t_index_range,
                     max_chunk_bytes=max_chunk_bytes)
