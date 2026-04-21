from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .._typing import BandpassProfileInput, FrequencyPathInput, FrequencyProfile, TimeProfileInput


@dataclass(frozen=True)
class _ResolvedSignalPath:
    """Normalized signal path arrays for synthetic rendering."""

    path: np.ndarray
    path_tt: np.ndarray
    dpath_tt: np.ndarray | None = None


def _resolve_bounding_indices(
    frame: Any,
    bounding_f_range: tuple[Any, Any] | None,
) -> tuple[int, int]:
    """Resolve a frequency bounding range into index bounds.

    Args:
        frame: Frame instance that defines the frequency axis.
        bounding_f_range: Optional minimum and maximum bounding frequencies.

    Returns:
        Inclusive lower and exclusive upper frequency indices.
    """
    if bounding_f_range is None:
        return 0, frame.fchans
    bounding_min = max(frame.get_index(bounding_f_range[0]), 0)
    bounding_max = min(frame.get_index(bounding_f_range[1]), frame.fchans)
    return bounding_min, bounding_max


def _get_restricted_fs(frame: Any,
                       *,
                       bounding_min: int,
                       bounding_max: int,
                       integrate_f_profile: bool = False,
                       f_subsamples: int = 10) -> tuple[np.ndarray, int]:
    """Build the working frequency axis for signal rendering.

    Args:
        frame: Frame instance that defines the frequency axis.
        bounding_min: Lower inclusive frequency index.
        bounding_max: Upper exclusive frequency index.
        integrate_f_profile: Whether to oversample in frequency before averaging.
        f_subsamples: Number of frequency subsamples per bin.

    Returns:
        Restricted frequency axis and its coarse-channel count.
    """
    restricted_fs = frame.fs[bounding_min:bounding_max]
    restricted_fchans = len(restricted_fs)
    if integrate_f_profile:
        f0 = restricted_fs[0]
        restricted_fs = np.linspace(f0,
                                    f0 + restricted_fchans * frame.df,
                                    restricted_fchans * f_subsamples,
                                    endpoint=False)
    return restricted_fs, restricted_fchans


def _normalize_t_profile(frame: Any,
                         restricted_fs: np.ndarray,
                         t_profile: TimeProfileInput,
                         *,
                         integrate_t_profile: bool = False,
                         t_subsamples: int = 10) -> np.ndarray:
    """Normalize a time profile into a time-frequency grid.

    Args:
        frame: Frame instance that defines the time axis.
        restricted_fs: Restricted frequency axis for mesh generation.
        t_profile: Callable, array, or scalar time profile.
        integrate_t_profile: Whether to oversample in time before averaging.
        t_subsamples: Number of time subsamples per bin.

    Returns:
        Time-profile mesh evaluated over the restricted frequency axis.

    Raises:
        TypeError: If the profile type is unsupported.
        ValueError: If an array-valued profile has the wrong shape.
    """
    if callable(t_profile):
        if integrate_t_profile:
            new_ts = np.linspace(0,
                                 frame.tchans * frame.dt,
                                 frame.tchans * t_subsamples,
                                 endpoint=False)
            y = t_profile(new_ts)
            if not isinstance(y, np.ndarray):
                y = np.repeat(y, frame.tchans * t_subsamples)
            t_profile = np.mean(np.reshape(y, (frame.tchans, t_subsamples)),
                                axis=1)
        else:
            t_profile = t_profile(frame.ts)
    elif isinstance(t_profile, (list, np.ndarray)):
        t_profile = np.array(t_profile)
        if t_profile.shape != frame.ts.shape:
            raise ValueError("Shape of t_profile array is {0} != {1}.".format(t_profile.shape,
                                                                               frame.ts.shape))
    elif isinstance(t_profile, (int, float)):
        t_profile = np.full(frame.tchans, t_profile)
    else:
        raise TypeError("t_profile is not a function, array, or float.")
    _, t_profile_tt = np.meshgrid(restricted_fs, t_profile)
    return t_profile_tt


def _normalize_bp_profile(
    frame: Any,
    restricted_fs: np.ndarray,
    bp_profile: BandpassProfileInput | None,
) -> np.ndarray:
    """Normalize a bandpass profile into a time-frequency grid.

    Args:
        frame: Frame instance that defines the time axis.
        restricted_fs: Restricted frequency axis for evaluation.
        bp_profile: Callable, array, scalar, or `None` bandpass profile.

    Returns:
        Bandpass profile mesh evaluated over the restricted frequency axis.

    Raises:
        TypeError: If the profile type is unsupported.
        ValueError: If an array-valued profile has the wrong shape.
    """
    if bp_profile is None:
        bp_profile = 1
    if callable(bp_profile):
        bp_profile = bp_profile(restricted_fs)
    elif isinstance(bp_profile, (list, np.ndarray)):
        bp_profile = np.array(bp_profile)
        if bp_profile.shape != restricted_fs.shape:
            raise ValueError("Shape of bp_profile array is {0} != {1}.".format(bp_profile.shape,
                                                                                restricted_fs.shape))
    elif isinstance(bp_profile, (int, float)):
        bp_profile = np.full(restricted_fs.shape, bp_profile)
    else:
        raise TypeError("bp_profile is not a function, array, or float.")
    bp_profile_ff, _ = np.meshgrid(bp_profile, frame.ts)
    return bp_profile_ff


def _normalize_path(frame: Any,
                    restricted_fs: np.ndarray,
                    path: FrequencyPathInput,
                    *,
                    integrate_path: bool = False,
                    doppler_smearing: bool = False,
                    t_subsamples: int = 10,
                    smearing_subsamples: int = 10) -> _ResolvedSignalPath:
    """Normalize a signal path into render-ready arrays.

    Args:
        frame: Frame instance that defines the time axis.
        restricted_fs: Restricted frequency axis for mesh generation.
        path: Callable, array, or scalar signal path.
        integrate_path: Whether to oversample in time before averaging the path.
        doppler_smearing: Whether to prepare differential path steps for
            Doppler smearing.
        t_subsamples: Number of time subsamples per bin.
        smearing_subsamples: Number of substeps used during Doppler smearing.

    Returns:
        Normalized signal path representation.

    Raises:
        TypeError: If the path type is unsupported.
        ValueError: If an array-valued path has the wrong shape.
    """
    # Generate one extra time sample for frequency-smearing calculations.
    tchans_eff = frame.tchans + int(doppler_smearing)

    if callable(path):
        if integrate_path:
            new_ts = np.linspace(0,
                                 tchans_eff * frame.dt,
                                 tchans_eff * t_subsamples,
                                 endpoint=False)
            f = path(new_ts)
            if not isinstance(f, np.ndarray):
                f = np.repeat(f, tchans_eff * t_subsamples)
            path = np.mean(np.reshape(f, (tchans_eff, t_subsamples)), axis=1)
        else:
            ts = frame.ts_ext if doppler_smearing else frame.ts
            path = path(ts)
    elif isinstance(path, (list, np.ndarray)):
        path = np.array(path)
        if doppler_smearing:
            if path.shape != frame.ts_ext.shape:
                raise ValueError(f"To Doppler smear power, must provide path array with {frame.tchans + 1} values")
        elif path.shape != frame.ts.shape:
            raise ValueError(f"Shape of path array is {path.shape} != {frame.ts.shape}.")
    elif isinstance(path, (int, float)):
        path = np.full(tchans_eff, path)
    else:
        raise TypeError("path is not a function, array, or float.")

    _, path_tt = np.meshgrid(restricted_fs, path[:frame.tchans])

    dpath_tt = None
    if doppler_smearing:
        dpath = np.diff(path) / smearing_subsamples
        _, dpath_tt = np.meshgrid(restricted_fs, dpath)

    return _ResolvedSignalPath(path=path, path_tt=path_tt, dpath_tt=dpath_tt)


def _render_signal(*,
                   ff: np.ndarray,
                   t_profile_tt: np.ndarray,
                   f_profile: FrequencyProfile,
                   bp_profile_ff: np.ndarray,
                   path_tt: np.ndarray,
                   doppler_smearing: bool = False,
                   dpath_tt: np.ndarray | None = None,
                   smearing_subsamples: int = 10) -> np.ndarray:
    """Render a synthetic signal onto a time-frequency grid.

    Args:
        ff: Time-frequency frequency mesh.
        t_profile_tt: Time-profile mesh.
        f_profile: Frequency profile callable.
        bp_profile_ff: Bandpass-profile mesh.
        path_tt: Signal-path mesh.
        doppler_smearing: Whether to numerically smear the signal in frequency.
        dpath_tt: Differential path steps for Doppler smearing.
        smearing_subsamples: Number of substeps used during Doppler smearing.

    Returns:
        Rendered signal array.
    """
    if not doppler_smearing:
        return t_profile_tt * f_profile(ff, path_tt) * bp_profile_ff

    signal = np.zeros(ff.shape)
    path_tt = np.array(path_tt, copy=True)
    for _ in range(smearing_subsamples):
        signal += (t_profile_tt * f_profile(ff, path_tt)
                   / smearing_subsamples * bp_profile_ff)
        path_tt += dpath_tt
    return signal


def _finalize_signal(frame: Any,
                     *,
                     signal: np.ndarray,
                     bounding_min: int,
                     bounding_max: int,
                     integrate_f_profile: bool = False,
                     restricted_fchans: int | None = None,
                     f_subsamples: int = 10) -> np.ndarray:
    """Finalize and add a rendered signal to a frame.

    Args:
        frame: Frame instance to update.
        signal: Rendered signal array.
        bounding_min: Lower inclusive frequency index.
        bounding_max: Upper exclusive frequency index.
        integrate_f_profile: Whether the signal must be averaged back in
            frequency.
        restricted_fchans: Number of coarse frequency bins in the restricted
            region.
        f_subsamples: Number of frequency subsamples per bin.

    Returns:
        Standalone signal frame aligned with the updated frame data.
    """
    if integrate_f_profile:
        signal = np.mean(np.reshape(signal,
                                    (frame.tchans,
                                     restricted_fchans,
                                     f_subsamples)),
                         axis=2)

    frame.data[:, bounding_min:bounding_max] += signal

    signal_frame = np.zeros(frame.shape)
    signal_frame[:, bounding_min:bounding_max] = signal
    return signal_frame
