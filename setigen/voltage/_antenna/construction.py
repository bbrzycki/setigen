from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from astropy import units as u

from setigen import unit_utils
from setigen.voltage import data_stream

if TYPE_CHECKING:
    import numpy as np
    from numpy.random import Generator


def _coerce_sample_rate(sample_rate: float | u.Quantity) -> float:
    """Coerce a sample-rate input into a float in Hz.

    Args:
        sample_rate: Sample rate as a scalar or astropy quantity.

    Returns:
        Sample rate in Hz.
    """
    return unit_utils.get_value(sample_rate, u.Hz)


def _coerce_fch1(fch1: float | u.Quantity) -> float:
    """Coerce a channel-start frequency input into a float in Hz.

    Args:
        fch1: First coarse-channel frequency as a scalar or astropy quantity.

    Returns:
        Frequency in Hz.
    """
    return unit_utils.get_value(fch1, u.Hz)


def _validate_num_pols(num_pols: int) -> int:
    """Validate the supported number of polarizations.

    Args:
        num_pols: Requested polarization count.

    Returns:
        Validated polarization count.

    Raises:
        ValueError: If ``num_pols`` is not 1 or 2.
    """
    if num_pols not in [1, 2]:
        raise ValueError("num_pols must be 1 or 2")
    return num_pols


def _build_polarization_streams(*,
                                sample_rate: float,
                                fch1: float,
                                ascending: bool,
                                t_start: float,
                                num_pols: int,
                                rng: Generator) -> tuple[data_stream.DataStream,
                                                         data_stream.DataStream | None,
                                                         list[data_stream.DataStream]]:
    """Create the polarization streams for a single antenna.

    Args:
        sample_rate: Stream sample rate in Hz.
        fch1: First coarse-channel frequency in Hz.
        ascending: Whether coarse channels ascend with index.
        t_start: Initial stream time in seconds.
        num_pols: Number of polarizations to instantiate.
        rng: Random-number generator used to seed child streams.

    Returns:
        Tuple of x-stream, optional y-stream, and a list of all created
        streams.
    """
    x_stream = data_stream.DataStream(sample_rate=sample_rate,
                                      fch1=fch1,
                                      ascending=ascending,
                                      t_start=t_start,
                                      seed=int(rng.integers(2**31)))
    streams = [x_stream]

    y_stream = None
    if num_pols == 2:
        y_stream = data_stream.DataStream(sample_rate=sample_rate,
                                          fch1=fch1,
                                          ascending=ascending,
                                          t_start=t_start,
                                          seed=int(rng.integers(2**31)))
        streams.append(y_stream)

    return x_stream, y_stream, streams


def _build_antennas(*,
                    num_antennas: int,
                    sample_rate: float,
                    fch1: float,
                    ascending: bool,
                    num_pols: int,
                    t_start: float,
                    rng: Generator,
                    antenna_cls: type,
                    delays: Sequence[int]) -> list:
    """Construct antenna objects and apply their integer delays.

    Args:
        num_antennas: Number of antennas to construct.
        sample_rate: Stream sample rate in Hz.
        fch1: First coarse-channel frequency in Hz.
        ascending: Whether coarse channels ascend with index.
        num_pols: Number of polarizations per antenna.
        t_start: Initial stream time in seconds.
        rng: Random-number generator used to seed child antennas.
        antenna_cls: Antenna class to instantiate.
        delays: Integer sample delays, one per antenna.

    Returns:
        List of constructed antennas with delays assigned.
    """
    antennas: list = []
    for i in range(num_antennas):
        antenna = antenna_cls(sample_rate=sample_rate,
                              fch1=fch1,
                              ascending=ascending,
                              num_pols=num_pols,
                              t_start=t_start,
                              seed=int(rng.integers(2**31)))
        antenna.delay = int(delays[i])
        antennas.append(antenna)
    return antennas


def _build_background_streams(*,
                              sample_rate: float,
                              fch1: float,
                              ascending: bool,
                              t_start: float,
                              num_pols: int,
                              rng: Generator,
                              antennas: Sequence) -> tuple[data_stream.BackgroundDataStream,
                                                           data_stream.BackgroundDataStream | None,
                                                           list[data_stream.BackgroundDataStream]]:
    """Create background streams shared across an antenna array.

    Args:
        sample_rate: Stream sample rate in Hz.
        fch1: First coarse-channel frequency in Hz.
        ascending: Whether coarse channels ascend with index.
        t_start: Initial stream time in seconds.
        num_pols: Number of polarizations per antenna.
        rng: Random-number generator used to seed child streams.
        antennas: Antenna objects whose foreground streams receive the
            background noise.

    Returns:
        Tuple of x-background stream, optional y-background stream, and a list
        of all created background streams.
    """
    bg_x = data_stream.BackgroundDataStream(sample_rate=sample_rate,
                                            fch1=fch1,
                                            ascending=ascending,
                                            t_start=t_start,
                                            seed=int(rng.integers(2**31)),
                                            antenna_streams=[antenna.x for antenna in antennas])
    bg_y = None
    streams = [bg_x]

    if num_pols == 2:
        bg_y = data_stream.BackgroundDataStream(sample_rate=sample_rate,
                                                fch1=fch1,
                                                ascending=ascending,
                                                t_start=t_start,
                                                seed=int(rng.integers(2**31)),
                                                antenna_streams=[antenna.y for antenna in antennas])
        streams.append(bg_y)

    return bg_x, bg_y, streams
