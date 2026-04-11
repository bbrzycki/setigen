from __future__ import annotations

import copy
from dataclasses import dataclass
import pathlib
import time

import numpy as np

from astropy import units as u
from astropy.time import Time

from blimpy import Waterfall

from .. import waterfall_utils
from .. import unit_utils


@dataclass(frozen=True)
class _SyntheticFrameSpec:
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


def _is_synthetic_init(*, fchans=None, tchans=None, data=None, kwargs=None):
    kwargs = {} if kwargs is None else kwargs
    return None not in [fchans, tchans] or "shape" in kwargs or data is not None


def _normalize_synthetic_init(*,
                              fchans=None,
                              tchans=None,
                              df=2.7939677238464355 * u.Hz,
                              dt=18.253611008 * u.s,
                              fch1=6 * u.GHz,
                              ascending=False,
                              data=None,
                              kwargs=None):
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


def _normalize_waterfall_init(*, waterfall, kwargs=None):
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


def _normalize_frame_init(*,
                          waterfall=None,
                          fchans=None,
                          tchans=None,
                          df=2.7939677238464355 * u.Hz,
                          dt=18.253611008 * u.s,
                          fch1=6 * u.GHz,
                          ascending=False,
                          data=None,
                          kwargs=None):
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


def _initialize_frame_from_spec(frame, spec):
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


def _attach_loaded_waterfall(frame, waterfall):
    if waterfall is None:
        return
    try:
        del waterfall.container.h5
    except AttributeError:
        pass
    frame.waterfall = copy.deepcopy(waterfall)
