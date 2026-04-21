import copy

import pytest
import numpy as np
from astropy import units as u

import setigen as stg


def test_data_stream_update_noise_preserves_time_state():
    stream = stg.voltage.DataStream(sample_rate=3 * u.GHz, seed=0)
    stream.add_noise(0, 1)
    stream.set_time(12.5)

    stream.update_noise(stats_calc_num_samples=128)

    assert stream.start_obs is True
    assert stream.t_start == pytest.approx(12.5)
    assert stream.noise_std > 0


def test_data_stream_add_time_and_complex_signal():
    stream = stg.voltage.DataStream(sample_rate=3 * u.GHz, seed=0)
    stream.add_time(1.5)
    assert stream.t_start == pytest.approx(1.5)
    assert stream.start_obs is True

    stream.add_signal(lambda ts: np.ones(len(ts), dtype=complex) * (1 + 2j))
    samples = stream.get_samples(8)

    assert np.iscomplexobj(samples)
    assert samples.shape == (8,)
    assert stream.start_obs is False


def test_background_data_stream_propagates_noise():
    antenna_streams = [stg.voltage.DataStream(seed=0), stg.voltage.DataStream(seed=1)]
    bg_stream = stg.voltage.BackgroundDataStream(antenna_streams=copy.deepcopy(antenna_streams),
                                                 seed=2)

    bg_stream.add_noise(0, 3)

    assert bg_stream.noise_std == pytest.approx(3)
    for stream in bg_stream.antenna_streams:
        assert stream.bg_noise_std == pytest.approx(3)


def test_antenna_set_time_propagates_to_streams():
    antenna = stg.voltage.Antenna(sample_rate=3 * u.GHz,
                                  fch1=6 * u.GHz,
                                  ascending=True,
                                  num_pols=2,
                                  seed=0)

    antenna.set_time(7.25)

    assert antenna.start_obs is True
    assert antenna.t_start == pytest.approx(7.25)
    assert antenna.x.t_start == pytest.approx(7.25)
    assert antenna.y.t_start == pytest.approx(7.25)


def test_multiantenna_default_delays_and_time_reset():
    antenna_array = stg.voltage.MultiAntennaArray(num_antennas=3,
                                                  sample_rate=3 * u.GHz,
                                                  fch1=6 * u.GHz,
                                                  ascending=False,
                                                  num_pols=2,
                                                  delays=None,
                                                  seed=0)

    assert np.array_equal(antenna_array.delays, np.zeros(3, dtype=int))
    assert antenna_array.max_delay == 0

    antenna_array.set_time(11.0)

    assert antenna_array.start_obs is True
    assert antenna_array.t_start == pytest.approx(11.0)
    for antenna in antenna_array.antennas:
        assert antenna.t_start == pytest.approx(11.0)
        assert antenna.bg_cache == [None, None]


def test_multiantenna_requires_num_samples_greater_than_max_delay():
    antenna_array = stg.voltage.MultiAntennaArray(num_antennas=2,
                                                  sample_rate=3 * u.GHz,
                                                  num_pols=1,
                                                  delays=[0, 4],
                                                  seed=0)

    with pytest.raises(ValueError, match="greater than the maximum antenna delay"):
        antenna_array.get_samples(4)
