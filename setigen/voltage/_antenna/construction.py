from __future__ import annotations

from astropy import units as u

from setigen import unit_utils
from setigen.voltage import data_stream


def _coerce_sample_rate(sample_rate):
    return unit_utils.get_value(sample_rate, u.Hz)


def _coerce_fch1(fch1):
    return unit_utils.get_value(fch1, u.Hz)


def _validate_num_pols(num_pols):
    if num_pols not in [1, 2]:
        raise ValueError("num_pols must be 1 or 2")
    return num_pols


def _build_polarization_streams(*,
                                sample_rate,
                                fch1,
                                ascending,
                                t_start,
                                num_pols,
                                rng):
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
                    num_antennas,
                    sample_rate,
                    fch1,
                    ascending,
                    num_pols,
                    t_start,
                    rng,
                    antenna_cls,
                    delays):
    antennas = []
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
                              sample_rate,
                              fch1,
                              ascending,
                              t_start,
                              num_pols,
                              rng,
                              antennas):
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
