from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class _FrequencyAxisKind(str, Enum):
    FMID = "fmid"
    FMIN = "fmin"
    FABS = "f"
    PIXELS = "pixels"


class _TimeAxisKind(str, Enum):
    SAME = "same"
    TREL = "trel"
    PIXELS = "pixels"


_PIXEL_AXIS_ALIASES = frozenset({"px", "bins", _FrequencyAxisKind.PIXELS.value})
_FREQUENCY_KIND_BY_NAME = {
    _FrequencyAxisKind.FMID.value: _FrequencyAxisKind.FMID,
    _FrequencyAxisKind.FMIN.value: _FrequencyAxisKind.FMIN,
    _FrequencyAxisKind.FABS.value: _FrequencyAxisKind.FABS,
}
_TIME_KIND_BY_NAME = {
    _TimeAxisKind.SAME.value: _TimeAxisKind.SAME,
    _TimeAxisKind.TREL.value: _TimeAxisKind.TREL,
}


@dataclass(frozen=True)
class _ResolvedAxisSpec:
    raw_ftype: str = "fmid"
    raw_ttype: str = "same"
    frequency_kind: _FrequencyAxisKind = _FrequencyAxisKind.FMID
    time_kind: _TimeAxisKind = _TimeAxisKind.SAME

    @classmethod
    def from_values(cls, ftype="fmid", ttype="same"):
        return cls(raw_ftype=ftype,
                   raw_ttype=ttype,
                   frequency_kind=_resolve_ftype(ftype),
                   time_kind=_resolve_ttype(ttype))

    @property
    def uses_frequency_units(self):
        return self.frequency_kind in {
            _FrequencyAxisKind.FMID,
            _FrequencyAxisKind.FMIN,
            _FrequencyAxisKind.FABS,
        }

    @property
    def uses_time_units(self):
        if self.time_kind is _TimeAxisKind.TREL:
            return True
        if self.time_kind is _TimeAxisKind.SAME:
            return self.uses_frequency_units
        return False


def _resolve_ftype(ftype):
    if isinstance(ftype, _FrequencyAxisKind):
        return ftype
    if ftype in _FREQUENCY_KIND_BY_NAME:
        return _FREQUENCY_KIND_BY_NAME[ftype]
    if ftype in _PIXEL_AXIS_ALIASES:
        return _FrequencyAxisKind.PIXELS
    return _FrequencyAxisKind.PIXELS


def _resolve_ttype(ttype):
    if isinstance(ttype, _TimeAxisKind):
        return ttype
    if ttype in _TIME_KIND_BY_NAME:
        return _TIME_KIND_BY_NAME[ttype]
    if ttype in _PIXEL_AXIS_ALIASES:
        return _TimeAxisKind.PIXELS
    return _TimeAxisKind.PIXELS


def _get_extent_units(frame):
    f_range = np.abs(frame.fmax - frame.fmin)
    if f_range > 2e9:
        return 1e9, "GHz"
    if f_range > 2e6:
        return 1e6, "MHz"
    if f_range > 2e3:
        return 1e3, "kHz"
    return 1, "Hz"


def _frequency_formatter(frame, ftype):
    axis_spec = _ResolvedAxisSpec.from_values(ftype=ftype)
    if axis_spec.frequency_kind in {
        _FrequencyAxisKind.FMID,
        _FrequencyAxisKind.FMIN,
    }:
        def formatter(x, pos):
            return x / _get_extent_units(frame)[0]
    else:
        def formatter(x, pos):
            return x / 1e6
    return formatter


def _get_frame_frequency_edges(frame, axis_spec):
    if axis_spec.frequency_kind is _FrequencyAxisKind.FMID:
        return frame.fmin - frame.fmid - frame.df / 2, frame.fmax - frame.fmid + frame.df / 2
    if axis_spec.frequency_kind is _FrequencyAxisKind.FMIN:
        return -frame.df / 2, frame.fmax - frame.fmin + frame.df / 2
    if axis_spec.frequency_kind is _FrequencyAxisKind.FABS:
        return frame.fmin - frame.df / 2, frame.fmax + frame.df / 2
    return -1 / 2, frame.fchans - 1 / 2


def _get_frame_time_edges(frame, axis_spec):
    if axis_spec.uses_time_units:
        return 0, frame.tchans * frame.dt
    return -1 / 2, frame.tchans - 1 / 2


def _get_frequency_axis_label(frame, axis_spec):
    if axis_spec.uses_frequency_units:
        units = _get_extent_units(frame)[1]
        if axis_spec.frequency_kind is _FrequencyAxisKind.FMID:
            return f"Relative Frequency ({units}) from {frame.fmid * 1e-6:.6f} MHz"
        if axis_spec.frequency_kind is _FrequencyAxisKind.FMIN:
            return f"Relative Frequency ({units}) from {frame.fmin * 1e-6:.6f} MHz"
        return "Frequency (MHz)"
    return f"Frequency ({axis_spec.raw_ftype})"


def _get_time_axis_label(axis_spec):
    if axis_spec.time_kind is _TimeAxisKind.SAME:
        if axis_spec.uses_frequency_units:
            return "Time (s)"
        return f"Time ({axis_spec.raw_ftype})"
    if axis_spec.time_kind is _TimeAxisKind.TREL:
        return "Time (s)"
    return f"Time ({axis_spec.raw_ttype})"


def _get_spectrum_x_values(spectrum, axis_spec):
    if axis_spec.frequency_kind is _FrequencyAxisKind.FMID:
        return spectrum.fs - spectrum.fmid
    if axis_spec.frequency_kind is _FrequencyAxisKind.FMIN:
        return spectrum.fs - spectrum.fmin
    if axis_spec.frequency_kind is _FrequencyAxisKind.FABS:
        return spectrum.fs
    return (spectrum.fs - spectrum.fs[0]) / spectrum.df


def _get_timeseries_x_values(timeseries, axis_spec):
    if axis_spec.uses_time_units:
        return timeseries.ts
    return (timeseries.ts - timeseries.ts[0]) / timeseries.dt
