from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np
from astropy.stats import sigma_clip
from . import utils
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
    if axis in [IntegrationAxis.FREQUENCY, IntegrationAxis.FREQUENCY.value, 1]:
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


def integrate(
    fr: Any,
    axis: IntegrationAxis | str | int = 't',
    mode: IntegrationMode | str = 'mean',
    normalize: bool = False,
    as_frame: bool = False,
) -> np.ndarray | Spectrum | TimeSeries:
    """Integrate frame data over time or frequency.

    Args:
        fr: Input frame or two-dimensional array.
        axis: Axis over which to integrate.
        mode: Integration mode.
        normalize: Whether to sigma-normalize the integrated result.
        as_frame: Whether to return a `Spectrum` or `TimeSeries` object.

    Returns:
        Integrated one-dimensional data or frame-like object.
    """
    # If `data` is a Frame object, just grab its data
    data = utils.array(fr)
    resolved_axis = _resolve_integration_axis(axis)
    resolved_mode = _resolve_integration_mode(mode)
    if resolved_axis is IntegrationAxis.FREQUENCY:
        # Time series
        axis = 1
    else:
        # Spectrum
        axis = 0
        
    if resolved_mode is IntegrationMode.SUM:
        data = np.sum(data, axis=axis, keepdims=True)
    else:
        data = np.mean(data, axis=axis, keepdims=True)
        
    if normalize:
        c_data = sigma_clip(data)
        data = (data - np.mean(c_data)) / np.std(c_data)

    if as_frame:
        if resolved_axis is IntegrationAxis.FREQUENCY:
            # Time series
            new_fr = TimeSeries(df=fr.df * fr.fchans,
                                dt=fr.dt,
                                fch1=fr.fmid,
                                ascending=fr.ascending,
                                data=data,
                                seed=fr.rng)
        else:
            # Spectrum
            new_fr = Spectrum(df=fr.df,
                              dt=fr.dt * fr.tchans,
                              fch1=fr.fch1,
                              ascending=fr.ascending,
                              data=data,
                              seed=fr.rng)
        return new_fr
    else:
        return data.flatten()


def spectrum(fr: Any, mode: IntegrationMode | str = "mean", normalize: bool = False) -> Spectrum:
    """Produce a `Spectrum` from a spectrogram frame.

    Args:
        fr: Input frame or array-like object.
        mode: Integration mode.
        normalize: Whether to sigma-normalize the result.

    Returns:
        Integrated spectrum.
    """
    return integrate(fr, axis=0, mode=mode, normalize=normalize, as_frame=True) 


def timeseries(fr: Any, mode: IntegrationMode | str = "mean", normalize: bool = False) -> TimeSeries:
    """Produce a `TimeSeries` from a spectrogram frame.

    Args:
        fr: Input frame or array-like object.
        mode: Integration mode.
        normalize: Whether to sigma-normalize the result.

    Returns:
        Integrated time series.
    """
    return integrate(fr, axis=1, mode=mode, normalize=normalize, as_frame=True)
