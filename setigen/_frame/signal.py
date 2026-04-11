from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class _ResolvedSignalPath:
    path: np.ndarray
    path_tt: np.ndarray
    dpath_tt: np.ndarray | None = None


def _resolve_bounding_indices(frame, bounding_f_range):
    if bounding_f_range is None:
        return 0, frame.fchans
    bounding_min = max(frame.get_index(bounding_f_range[0]), 0)
    bounding_max = min(frame.get_index(bounding_f_range[1]), frame.fchans)
    return bounding_min, bounding_max


def _get_restricted_fs(frame,
                       *,
                       bounding_min,
                       bounding_max,
                       integrate_f_profile=False,
                       f_subsamples=10):
    restricted_fs = frame.fs[bounding_min:bounding_max]
    restricted_fchans = len(restricted_fs)
    if integrate_f_profile:
        f0 = restricted_fs[0]
        restricted_fs = np.linspace(f0,
                                    f0 + restricted_fchans * frame.df,
                                    restricted_fchans * f_subsamples,
                                    endpoint=False)
    return restricted_fs, restricted_fchans


def _normalize_t_profile(frame,
                         restricted_fs,
                         t_profile,
                         *,
                         integrate_t_profile=False,
                         t_subsamples=10):
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


def _normalize_bp_profile(frame, restricted_fs, bp_profile):
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


def _normalize_path(frame,
                    restricted_fs,
                    path,
                    *,
                    integrate_path=False,
                    doppler_smearing=False,
                    t_subsamples=10,
                    smearing_subsamples=10):
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
                   ff,
                   t_profile_tt,
                   f_profile,
                   bp_profile_ff,
                   path_tt,
                   doppler_smearing=False,
                   dpath_tt=None,
                   smearing_subsamples=10):
    if not doppler_smearing:
        return t_profile_tt * f_profile(ff, path_tt) * bp_profile_ff

    signal = np.zeros(ff.shape)
    path_tt = np.array(path_tt, copy=True)
    for _ in range(smearing_subsamples):
        signal += (t_profile_tt * f_profile(ff, path_tt)
                   / smearing_subsamples * bp_profile_ff)
        path_tt += dpath_tt
    return signal


def _finalize_signal(frame,
                     *,
                     signal,
                     bounding_min,
                     bounding_max,
                     integrate_f_profile=False,
                     restricted_fchans=None,
                     f_subsamples=10):
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
