from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..frame import Frame
    from ..spectrum import Spectrum
    from ..timeseries import TimeSeries


class _FrequencyAxisKind(str, Enum):
    """Supported frequency-axis interpretations for frame plots."""

    FMID = "fmid"
    FMIN = "fmin"
    FABS = "f"
    PIXELS = "pixels"


class _TimeAxisKind(str, Enum):
    """Supported time-axis interpretations for frame and series plots."""

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
    """Resolved plot-axis configuration after alias normalization."""

    raw_ftype: str = "fmid"
    raw_ttype: str = "same"
    frequency_kind: _FrequencyAxisKind = _FrequencyAxisKind.FMID
    time_kind: _TimeAxisKind = _TimeAxisKind.SAME

    @classmethod
    def from_values(cls,
                    ftype: str | _FrequencyAxisKind = "fmid",
                    ttype: str | _TimeAxisKind = "same") -> _ResolvedAxisSpec:
        """Construct a resolved axis specification from user inputs.

        Args:
            ftype: Requested frequency-axis mode or alias.
            ttype: Requested time-axis mode or alias.

        Returns:
            Normalized axis-specification object.
        """
        return cls(raw_ftype=ftype,
                   raw_ttype=ttype,
                   frequency_kind=_resolve_ftype(ftype),
                   time_kind=_resolve_ttype(ttype))

    @property
    def uses_frequency_units(self) -> bool:
        """Whether the frequency axis should be labeled in physical units."""
        return self.frequency_kind in {
            _FrequencyAxisKind.FMID,
            _FrequencyAxisKind.FMIN,
            _FrequencyAxisKind.FABS,
        }

    @property
    def uses_time_units(self) -> bool:
        """Whether the time axis should be labeled in physical units."""
        if self.time_kind is _TimeAxisKind.TREL:
            return True
        if self.time_kind is _TimeAxisKind.SAME:
            return self.uses_frequency_units
        return False


def _resolve_ftype(ftype: str | _FrequencyAxisKind) -> _FrequencyAxisKind:
    """Resolve a user-facing frequency-axis selector into an enum value.

    Args:
        ftype: Requested frequency-axis selector or alias.

    Returns:
        Normalized frequency-axis enum.
    """
    if isinstance(ftype, _FrequencyAxisKind):
        return ftype
    if ftype in _FREQUENCY_KIND_BY_NAME:
        return _FREQUENCY_KIND_BY_NAME[ftype]
    if ftype in _PIXEL_AXIS_ALIASES:
        return _FrequencyAxisKind.PIXELS
    return _FrequencyAxisKind.PIXELS


def _resolve_ttype(ttype: str | _TimeAxisKind) -> _TimeAxisKind:
    """Resolve a user-facing time-axis selector into an enum value.

    Args:
        ttype: Requested time-axis selector or alias.

    Returns:
        Normalized time-axis enum.
    """
    if isinstance(ttype, _TimeAxisKind):
        return ttype
    if ttype in _TIME_KIND_BY_NAME:
        return _TIME_KIND_BY_NAME[ttype]
    if ttype in _PIXEL_AXIS_ALIASES:
        return _TimeAxisKind.PIXELS
    return _TimeAxisKind.PIXELS


def _get_extent_units(frame: Frame) -> tuple[float, str]:
    """Choose a frequency scale factor and display unit for a frame.

    Args:
        frame: Input frame whose bandwidth determines the display unit.

    Returns:
        Tuple of scale factor and unit label.
    """
    f_range = np.abs(frame.fmax - frame.fmin)
    if f_range > 2e9:
        return 1e9, "GHz"
    if f_range > 2e6:
        return 1e6, "MHz"
    if f_range > 2e3:
        return 1e3, "kHz"
    return 1, "Hz"


def _frequency_formatter(frame: Frame,
                         ftype: str | _FrequencyAxisKind) -> Callable[[float, float], float]:
    """Create a tick formatter for frequency axes.

    Args:
        frame: Input frame being plotted.
        ftype: Requested frequency-axis mode.

    Returns:
        Callable suitable for use as a matplotlib tick formatter.
    """
    axis_spec = _ResolvedAxisSpec.from_values(ftype=ftype)
    if axis_spec.frequency_kind in {
        _FrequencyAxisKind.FMID,
        _FrequencyAxisKind.FMIN,
    }:
        def formatter(x: float, pos: float) -> float:
            return x / _get_extent_units(frame)[0]
    else:
        def formatter(x: float, pos: float) -> float:
            return x / 1e6
    return formatter


def _get_frame_frequency_edges(frame: Frame,
                               axis_spec: _ResolvedAxisSpec) -> tuple[float, float]:
    """Get the frequency-axis extent for an image plot.

    Args:
        frame: Input frame being plotted.
        axis_spec: Resolved axis specification.

    Returns:
        Lower and upper plot edges for the frequency axis.
    """
    if axis_spec.frequency_kind is _FrequencyAxisKind.FMID:
        return frame.fmin - frame.fmid - frame.df / 2, frame.fmax - frame.fmid + frame.df / 2
    if axis_spec.frequency_kind is _FrequencyAxisKind.FMIN:
        return -frame.df / 2, frame.fmax - frame.fmin + frame.df / 2
    if axis_spec.frequency_kind is _FrequencyAxisKind.FABS:
        return frame.fmin - frame.df / 2, frame.fmax + frame.df / 2
    return -1 / 2, frame.fchans - 1 / 2


def _get_frame_time_edges(frame: Frame,
                          axis_spec: _ResolvedAxisSpec) -> tuple[float, float]:
    """Get the time-axis extent for an image plot.

    Args:
        frame: Input frame being plotted.
        axis_spec: Resolved axis specification.

    Returns:
        Lower and upper plot edges for the time axis.
    """
    if axis_spec.uses_time_units:
        return 0, frame.tchans * frame.dt
    return -1 / 2, frame.tchans - 1 / 2


def _get_frequency_axis_label(frame: Frame,
                              axis_spec: _ResolvedAxisSpec) -> str:
    """Build the display label for a frequency axis.

    Args:
        frame: Input frame being plotted.
        axis_spec: Resolved axis specification.

    Returns:
        Frequency-axis label string.
    """
    if axis_spec.uses_frequency_units:
        units = _get_extent_units(frame)[1]
        if axis_spec.frequency_kind is _FrequencyAxisKind.FMID:
            return f"Relative Frequency ({units}) from {frame.fmid * 1e-6:.6f} MHz"
        if axis_spec.frequency_kind is _FrequencyAxisKind.FMIN:
            return f"Relative Frequency ({units}) from {frame.fmin * 1e-6:.6f} MHz"
        return "Frequency (MHz)"
    return f"Frequency ({axis_spec.raw_ftype})"


def _get_time_axis_label(axis_spec: _ResolvedAxisSpec) -> str:
    """Build the display label for a time axis.

    Args:
        axis_spec: Resolved axis specification.

    Returns:
        Time-axis label string.
    """
    if axis_spec.time_kind is _TimeAxisKind.SAME:
        if axis_spec.uses_frequency_units:
            return "Time (s)"
        return f"Time ({axis_spec.raw_ftype})"
    if axis_spec.time_kind is _TimeAxisKind.TREL:
        return "Time (s)"
    return f"Time ({axis_spec.raw_ttype})"


def _get_spectrum_x_values(spectrum: Spectrum,
                           axis_spec: _ResolvedAxisSpec) -> np.ndarray:
    """Get x-axis values for a spectrum plot.

    Args:
        spectrum: Spectrum being plotted.
        axis_spec: Resolved axis specification.

    Returns:
        Array of x-axis values for plotting.
    """
    if axis_spec.frequency_kind is _FrequencyAxisKind.FMID:
        return spectrum.fs - spectrum.fmid
    if axis_spec.frequency_kind is _FrequencyAxisKind.FMIN:
        return spectrum.fs - spectrum.fmin
    if axis_spec.frequency_kind is _FrequencyAxisKind.FABS:
        return spectrum.fs
    return (spectrum.fs - spectrum.fs[0]) / spectrum.df


def _get_timeseries_x_values(timeseries: TimeSeries,
                             axis_spec: _ResolvedAxisSpec) -> np.ndarray:
    """Get x-axis values for a time-series plot.

    Args:
        timeseries: Time series being plotted.
        axis_spec: Resolved axis specification.

    Returns:
        Array of x-axis values for plotting.
    """
    if axis_spec.uses_time_units:
        return timeseries.ts
    return (timeseries.ts - timeseries.ts[0]) / timeseries.dt
