from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .signal import (
    _evaluate_path_values,
    _finalize_signal,
    _get_restricted_fs,
    _normalize_bp_profile,
    _normalize_path,
    _normalize_t_profile,
    _render_signal,
    _resolve_auto_bounding_range,
    _resolve_bounding_indices,
)


@dataclass(frozen=True)
class FileBackedSignalResult:
    """Summary of a memory-bounded file-backed signal injection."""

    shape: tuple[int, int]
    frequency_slice: slice
    time_chunks: int
    max_chunk_shape: tuple[int, int]


class _ChunkFrameView:
    """Small frame-like view used to reuse signal rendering helpers per chunk."""

    def __init__(self,
                 source: Any,
                 *,
                 t_start_index: int,
                 tchans: int,
                 f_start_index: int,
                 f_stop_index: int,
                 data: np.ndarray) -> None:
        """Create a chunk view with frame-like axes and data.

        Args:
            source: Full source frame.
            t_start_index: Absolute time-bin start index for this chunk.
            tchans: Number of time bins in this chunk.
            f_start_index: Absolute frequency-channel start index.
            f_stop_index: Absolute frequency-channel stop index.
            data: Chunk data in internal `Frame` orientation.
        """
        self.df = source.df
        self.dt = source.dt
        self.ascending = source.ascending
        self.t_start = source.t_start + t_start_index * source.dt
        self.source_name = source.source_name
        self.fchans = f_stop_index - f_start_index
        self.tchans = tchans
        self.shape = (tchans, self.fchans)
        self.fs = source.fs[f_start_index:f_stop_index]
        if self.ascending:
            self.fch1 = self.fs[0]
        else:
            self.fch1 = self.fs[-1]
        self.fmin = self.fs[0]
        self.fmax = self.fs[-1]
        self.ts = np.linspace(0, self.tchans * self.dt, self.tchans, endpoint=False)
        self.data = data

    @property
    def time_edges(self) -> np.ndarray:
        """Return chunk time-bin edges in seconds."""
        return np.linspace(0, self.tchans * self.dt, self.tchans + 1, endpoint=True)

    @property
    def ts_ext(self) -> np.ndarray:
        """Return chunk time-bin edges for legacy signal helpers."""
        return self.time_edges

    def get_index(self, frequency: Any) -> np.ndarray:
        """Convert frequency to the closest local chunk channel index.

        Args:
            frequency: Frequency or array of frequencies in Hz.

        Returns:
            Closest local frequency-channel index or indices.
        """
        return np.round((frequency - self.fmin) / self.df).astype(int)


def _slice_time_input(value: Any,
                      *,
                      t_start: int,
                      t_stop: int,
                      full_tchans: int,
                      doppler_smearing: bool = False) -> Any:
    """Slice full-observation time/path arrays for one chunk.

    Args:
        value: Candidate time-dependent profile or path.
        t_start: Inclusive chunk time-bin start index.
        t_stop: Exclusive chunk time-bin stop index.
        full_tchans: Number of time bins in the full frame.
        doppler_smearing: Whether path arrays need one extra edge sample.

    Returns:
        Sliced array when `value` matches the full time axis, otherwise `value`.
    """
    if not isinstance(value, (list, np.ndarray)):
        return value
    array = np.asarray(value)
    expected = full_tchans + int(doppler_smearing)
    if array.shape == (expected,):
        return array[t_start:t_stop + int(doppler_smearing)]
    return value


def _evaluate_t_profile_values(frame: Any,
                               t_profile: Any,
                               *,
                               integrate_t_profile: bool = False,
                               t_subsamples: int = 10,
                               t_offset: float = 0) -> Any:
    """Evaluate a callable time profile over the full frame once.

    Args:
        frame: Full frame defining the time axis.
        t_profile: Time profile input.
        integrate_t_profile: Whether to average over time subsamples.
        t_subsamples: Number of time subsamples per bin.
        t_offset: Time offset applied before evaluation.

    Returns:
        Full-frame time-profile array, or the original non-callable input.
    """
    if not callable(t_profile):
        return t_profile
    if integrate_t_profile:
        ts = np.linspace(0,
                         frame.tchans * frame.dt,
                         frame.tchans * t_subsamples,
                         endpoint=False) + t_offset
        values = t_profile(ts)
        if not isinstance(values, np.ndarray):
            values = np.repeat(values, frame.tchans * t_subsamples)
        return np.mean(np.reshape(values, (frame.tchans, t_subsamples)), axis=1)

    values = t_profile(frame.ts + t_offset)
    if not isinstance(values, np.ndarray):
        values = np.full(frame.tchans, values)
    return values


def _choose_chunk_tchans(frame: Any,
                         *,
                         affected_fchans: int,
                         max_chunk_bytes: int | None,
                         chunk_tchans: int | None) -> int:
    """Choose a time-chunk length from explicit or byte-budget inputs.

    Args:
        frame: File-backed frame being patched.
        affected_fchans: Number of affected frequency channels.
        max_chunk_bytes: Optional memory budget for one chunk.
        chunk_tchans: Optional explicit time-bin count per chunk.

    Returns:
        Number of time bins to process per chunk.
    """
    if chunk_tchans is not None:
        if chunk_tchans < 1:
            raise ValueError("chunk_tchans must be at least 1")
        return min(chunk_tchans, frame.tchans)

    if max_chunk_bytes is None:
        max_chunk_bytes = getattr(frame, "_max_chunk_bytes", 256 * 1024 * 1024)
    if max_chunk_bytes < 1:
        raise ValueError("max_chunk_bytes must be positive")

    backend = frame._file_backend
    itemsize = np.dtype(getattr(backend, "dtype", np.float32)).itemsize
    working_arrays = 4
    row_bytes = max(affected_fchans, 1) * itemsize * working_arrays
    return max(1, min(frame.tchans, int(max_chunk_bytes // max(row_bytes, 1))))


def add_signal_to_file_backed_frame(
    frame: Any,
    *,
    path: Any,
    t_profile: Any,
    f_profile: Any,
    bp_profile: Any = None,
    bounding_f_range: tuple[Any, Any] | None = None,
    integrate_path: bool = False,
    integrate_t_profile: bool = False,
    integrate_f_profile: bool = False,
    doppler_smearing: bool = False,
    t_subsamples: int = 10,
    f_subsamples: int = 10,
    smearing_subsamples: int = 10,
    t_offset: float = 0,
    auto_bounding: bool = False,
    truncate_below: float | None = None,
    max_chunk_bytes: int | None = None,
    chunk_tchans: int | None = None,
) -> FileBackedSignalResult:
    """Patch a writable file-backed frame immediately in bounded chunks.

    Args:
        frame: Writable file-backed frame.
        path: Signal frequency path.
        t_profile: Signal time profile.
        f_profile: Signal frequency profile.
        bp_profile: Optional bandpass profile.
        bounding_f_range: Optional frequency range limiting rendering.
        integrate_path: Whether to integrate the path within time bins.
        integrate_t_profile: Whether to integrate the time profile.
        integrate_f_profile: Whether to integrate the frequency profile.
        doppler_smearing: Whether to smear power across drift within bins.
        t_subsamples: Number of time subsamples.
        f_subsamples: Number of frequency subsamples.
        smearing_subsamples: Number of Doppler-smearing substeps.
        t_offset: Time offset for callable path/profile evaluation.
        auto_bounding: Whether to infer bounds for known profiles.
        truncate_below: Optional relative cutoff for infinite-support profiles.
        max_chunk_bytes: Optional memory budget per chunk.
        chunk_tchans: Optional explicit time bins per chunk.

    Returns:
        Summary of the patched region and chunking.
    """
    backend = frame._file_backend
    if not backend.writable:
        raise OSError(
            "This file-backed frame is read-only. Use Frame.open_copy(...) or "
            "Frame.open(..., mode='r+', allow_inplace=True) before add_signal()."
        )

    if doppler_smearing and smearing_subsamples < 1:
        raise ValueError("smearing_subsamples must be at least 1 when doppler_smearing=True")

    if auto_bounding and bounding_f_range is None:
        bounding_f_range, path = _resolve_auto_bounding_range(
            frame,
            path,
            f_profile,
            integrate_path=integrate_path,
            integrate_f_profile=integrate_f_profile,
            doppler_smearing=doppler_smearing,
            t_subsamples=t_subsamples,
            t_offset=t_offset,
            truncate_below=truncate_below,
        )

    if callable(path):
        path = _evaluate_path_values(frame,
                                     path,
                                     integrate_path=integrate_path,
                                     doppler_smearing=doppler_smearing,
                                     t_subsamples=t_subsamples,
                                     t_offset=t_offset)
    if callable(t_profile):
        t_profile = _evaluate_t_profile_values(frame,
                                               t_profile,
                                               integrate_t_profile=integrate_t_profile,
                                               t_subsamples=t_subsamples,
                                               t_offset=t_offset)

    bounding_min, bounding_max = _resolve_bounding_indices(frame, bounding_f_range)
    if bounding_min >= bounding_max:
        return FileBackedSignalResult(shape=frame.shape,
                                      frequency_slice=slice(bounding_min, bounding_max),
                                      time_chunks=0,
                                      max_chunk_shape=(0, 0))

    affected_fchans = bounding_max - bounding_min
    chunk_len = _choose_chunk_tchans(frame,
                                     affected_fchans=affected_fchans,
                                     max_chunk_bytes=max_chunk_bytes,
                                     chunk_tchans=chunk_tchans)
    chunks = 0
    max_chunk_shape = (0, affected_fchans)

    for chunk_start in range(0, frame.tchans, chunk_len):
        chunk_stop = min(frame.tchans, chunk_start + chunk_len)
        chunk_data = backend.read_region(chunk_start,
                                         chunk_stop,
                                         bounding_min,
                                         bounding_max)
        chunk_frame = _ChunkFrameView(frame,
                                      t_start_index=chunk_start,
                                      tchans=chunk_stop - chunk_start,
                                      f_start_index=bounding_min,
                                      f_stop_index=bounding_max,
                                      data=chunk_data)
        local_path = _slice_time_input(path,
                                       t_start=chunk_start,
                                       t_stop=chunk_stop,
                                       full_tchans=frame.tchans,
                                       doppler_smearing=doppler_smearing)
        local_t_profile = _slice_time_input(t_profile,
                                            t_start=chunk_start,
                                            t_stop=chunk_stop,
                                            full_tchans=frame.tchans)

        restricted_fs, restricted_fchans = _get_restricted_fs(
            chunk_frame,
            bounding_min=0,
            bounding_max=chunk_frame.fchans,
            integrate_f_profile=integrate_f_profile,
            f_subsamples=f_subsamples,
        )
        ff, _ = np.meshgrid(restricted_fs, chunk_frame.ts)

        local_t_offset = t_offset + chunk_start * frame.dt
        t_profile_tt = _normalize_t_profile(
            chunk_frame,
            restricted_fs,
            local_t_profile,
            integrate_t_profile=integrate_t_profile,
            t_subsamples=t_subsamples,
            t_offset=local_t_offset,
        )
        resolved_path = _normalize_path(
            chunk_frame,
            restricted_fs,
            local_path,
            integrate_path=integrate_path,
            doppler_smearing=doppler_smearing,
            t_subsamples=t_subsamples,
            smearing_subsamples=smearing_subsamples,
            t_offset=local_t_offset,
        )
        bp_profile_ff = _normalize_bp_profile(chunk_frame, restricted_fs, bp_profile)
        signal = _render_signal(ff=ff,
                                t_profile_tt=t_profile_tt,
                                f_profile=f_profile,
                                bp_profile_ff=bp_profile_ff,
                                path_tt=resolved_path.path_tt,
                                doppler_smearing=doppler_smearing,
                                dpath_tt=resolved_path.dpath_tt,
                                smearing_subsamples=smearing_subsamples)
        _finalize_signal(chunk_frame,
                         signal=signal,
                         bounding_min=0,
                         bounding_max=chunk_frame.fchans,
                         integrate_f_profile=integrate_f_profile,
                         restricted_fchans=restricted_fchans,
                         f_subsamples=f_subsamples)
        backend.write_region(chunk_start, bounding_min, chunk_frame.data)
        chunks += 1
        max_chunk_shape = (
            max(max_chunk_shape[0], chunk_frame.data.shape[0]),
            max(max_chunk_shape[1], chunk_frame.data.shape[1]),
        )

    backend.flush()
    return FileBackedSignalResult(shape=frame.shape,
                                  frequency_slice=slice(bounding_min, bounding_max),
                                  time_chunks=chunks,
                                  max_chunk_shape=max_chunk_shape)
