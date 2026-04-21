from __future__ import annotations

import copy
from dataclasses import dataclass
import pathlib
import time
from typing import Any

import numpy as np

from astropy import units as u
from astropy.time import Time

from blimpy import Waterfall

from .. import waterfall_utils
from .. import unit_utils


@dataclass(frozen=True)
class _SyntheticFrameSpec:
    """Normalized specification for creating a synthetic frame."""

    df: float
    dt: float
    fch1: float
    ascending: bool
    shape: tuple[int, int]
    data: np.ndarray
    t_start: float
    source_name: str


@dataclass(frozen=True)
class _WaterfallLoadSpec:
    """Normalized specification for loading a frame from a waterfall."""

    waterfall: Waterfall
    header: object
    df: float
    dt: float
    fch1: float
    ascending: bool
    shape: tuple[int, int]
    data: np.ndarray
    t_start: float
    source_name: str


def _is_synthetic_init(
    *,
    fchans: int | None = None,
    tchans: int | None = None,
    data: np.ndarray | None = None,
    kwargs: dict[str, Any] | None = None,
) -> bool:
    """Return whether a frame should be initialized synthetically.

    Args:
        fchans: Number of frequency channels.
        tchans: Number of time channels.
        data: Optional preloaded data array.
        kwargs: Additional frame-construction keyword arguments.

    Returns:
        Whether the supplied initialization parameters describe a synthetic frame.
    """
    kwargs = {} if kwargs is None else kwargs
    return None not in [fchans, tchans] or "shape" in kwargs or data is not None


def _normalize_synthetic_init(
    *,
    fchans: int | None = None,
    tchans: int | None = None,
    df: Any = 2.7939677238464355 * u.Hz,
    dt: Any = 18.253611008 * u.s,
    fch1: Any = 6 * u.GHz,
    ascending: bool = False,
    data: np.ndarray | None = None,
    kwargs: dict[str, Any] | None = None,
) -> _SyntheticFrameSpec:
    """Normalize synthetic frame-construction inputs.

    Args:
        fchans: Number of frequency channels.
        tchans: Number of time channels.
        df: Frequency resolution.
        dt: Time resolution.
        fch1: First-channel frequency.
        ascending: Whether the frequency axis is ascending.
        data: Optional preloaded frame data.
        kwargs: Additional frame-construction keyword arguments.

    Returns:
        Normalized synthetic frame specification.

    Raises:
        ValueError: If the supplied data shape does not match the requested frame
            shape.
    """
    kwargs = {} if kwargs is None else kwargs

    normalized_df = unit_utils.get_value(abs(df), u.Hz)
    normalized_dt = unit_utils.get_value(dt, u.s)
    normalized_fch1 = unit_utils.get_value(fch1, u.Hz)

    mjd = kwargs.get("mjd")
    if mjd is not None:
        t_start = Time(mjd, format="mjd").unix
    else:
        t_start = kwargs.get("t_start", time.time())
    source_name = kwargs.get("source_name", "Synthetic")

    if "shape" in kwargs:
        shape = tuple(kwargs["shape"])
    elif data is not None:
        shape = data.shape
    else:
        shape = (int(unit_utils.get_value(tchans, u.pixel)),
                 int(unit_utils.get_value(fchans, u.pixel)))

    if data is not None:
        if data.shape != shape:
            raise ValueError(f"Data shape {data.shape} does not match frame shape {shape}.")
        frame_data = np.copy(data)
    else:
        frame_data = np.zeros(shape)

    return _SyntheticFrameSpec(df=normalized_df,
                               dt=normalized_dt,
                               fch1=normalized_fch1,
                               ascending=ascending,
                               shape=shape,
                               data=frame_data,
                               t_start=t_start,
                               source_name=source_name)


def _normalize_waterfall_init(
    *,
    waterfall: str | pathlib.PurePath | Waterfall,
    kwargs: dict[str, Any] | None = None,
) -> _WaterfallLoadSpec:
    """Normalize frame-construction inputs sourced from a waterfall.

    Args:
        waterfall: Waterfall object or path to a waterfall-backed file.
        kwargs: Additional loader keyword arguments.

    Returns:
        Normalized waterfall-backed frame specification.

    Raises:
        FileNotFoundError: If the supplied waterfall object type is unsupported.
    """
    kwargs = {} if kwargs is None else kwargs

    if isinstance(waterfall, pathlib.PurePath):
        waterfall = str(waterfall)
    if isinstance(waterfall, str):
        waterfall = Waterfall(waterfall,
                              f_start=kwargs.get("f_start"),
                              f_stop=kwargs.get("f_stop"))
    elif not isinstance(waterfall, Waterfall):
        raise FileNotFoundError(f"Unsupported data type: {type(waterfall)}")

    header = waterfall.header
    tchans, _, fchans = waterfall.container.selection_shape
    shape = (tchans, fchans)

    df = unit_utils.cast_value(abs(header["foff"]), u.MHz).to(u.Hz).value
    dt = unit_utils.get_value(header["tsamp"], u.s)

    ascending = header["foff"] > 0
    if ascending:
        fch1 = waterfall.container.f_start
    else:
        fch1 = waterfall.container.f_stop
    fch1 = unit_utils.cast_value(fch1, u.MHz).to(u.Hz).value

    t_start = Time(header["tstart"], format="mjd").unix
    source_name = header["source_name"]

    data = waterfall_utils.get_data(waterfall)
    if not ascending:
        data = data[:, ::-1]

    return _WaterfallLoadSpec(waterfall=waterfall,
                              header=header,
                              df=df,
                              dt=dt,
                              fch1=fch1,
                              ascending=ascending,
                              shape=shape,
                              data=data,
                              t_start=t_start,
                              source_name=source_name)


def _normalize_frame_init(
    *,
    waterfall: str | pathlib.PurePath | Waterfall | None = None,
    fchans: int | None = None,
    tchans: int | None = None,
    df: Any = 2.7939677238464355 * u.Hz,
    dt: Any = 18.253611008 * u.s,
    fch1: Any = 6 * u.GHz,
    ascending: bool = False,
    data: np.ndarray | None = None,
    kwargs: dict[str, Any] | None = None,
) -> _SyntheticFrameSpec | _WaterfallLoadSpec:
    """Normalize frame-construction inputs from either supported source.

    Args:
        waterfall: Waterfall object or path to a waterfall-backed file.
        fchans: Number of frequency channels.
        tchans: Number of time channels.
        df: Frequency resolution.
        dt: Time resolution.
        fch1: First-channel frequency.
        ascending: Whether the frequency axis is ascending.
        data: Optional preloaded frame data.
        kwargs: Additional construction keyword arguments.

    Returns:
        Normalized frame specification.

    Raises:
        ValueError: If neither synthetic dimensions nor a waterfall source is
            provided.
    """
    kwargs = {} if kwargs is None else kwargs
    if _is_synthetic_init(fchans=fchans,
                          tchans=tchans,
                          data=data,
                          kwargs=kwargs):
        return _normalize_synthetic_init(fchans=fchans,
                                         tchans=tchans,
                                         df=df,
                                         dt=dt,
                                         fch1=fch1,
                                         ascending=ascending,
                                         data=data,
                                         kwargs=kwargs)
    if waterfall is not None:
        return _normalize_waterfall_init(waterfall=waterfall, kwargs=kwargs)
    raise ValueError("Frame must be provided dimensions or an existing filterbank file.")


def _initialize_frame_from_spec(
    frame: Any,
    spec: _SyntheticFrameSpec | _WaterfallLoadSpec,
) -> None:
    """Populate a frame object from a normalized construction spec.

    Args:
        frame: Frame instance to populate.
        spec: Normalized synthetic or waterfall-backed specification.
    """
    frame.df = spec.df
    frame.dt = spec.dt
    frame.fch1 = spec.fch1
    frame.ascending = spec.ascending
    frame.t_start = spec.t_start
    frame.source_name = spec.source_name
    frame.shape = spec.shape
    frame.tchans, frame.fchans = spec.shape
    frame.data = spec.data

    if isinstance(spec, _WaterfallLoadSpec):
        frame.waterfall = spec.waterfall
        frame.header = spec.header
    else:
        frame.waterfall = None
        frame.header = None


def _attach_loaded_waterfall(frame: Any, waterfall: Waterfall | None) -> None:
    """Attach a deepcopy of a loaded waterfall to a frame.

    Args:
        frame: Frame instance to update.
        waterfall: Loaded waterfall to attach, if one exists.
    """
    if waterfall is None:
        return
    try:
        del waterfall.container.h5
    except AttributeError:
        pass
    frame.waterfall = copy.deepcopy(waterfall)
