import pytest
import numpy as np
from numpy.testing import assert_allclose

import setigen as stg
from setigen.voltage.polyphase_filterbank import pfb_frontend, pfb_frontend_reference


def test_filterbank():
    filterbank = stg.voltage.PolyphaseFilterbank(num_taps=8,
                                                 num_branches=1024)
    response = filterbank.get_response(fftlength=512)
    assert np.max(response) / np.mean(response) == pytest.approx(1.1101855941416805)
    assert len(response) == 256

    assert len(filterbank.tile_response(num_chans=64, fftlength=512)) == 64 * 512


def test_filterbank_rejects_response_fftlength_not_multiple_of_taps():
    filterbank = stg.voltage.PolyphaseFilterbank(num_taps=8,
                                                 num_branches=64)

    with pytest.raises(ValueError, match="must be a multiple"):
        filterbank.get_response(fftlength=10)


def test_pfb_voltages():
    antenna = stg.voltage.Antenna(sample_rate=3e9, 
                                  fch1=6e9,
                                  ascending=False,
                                  num_pols=1,
                                  seed=0)
    antenna.x.add_noise(0, 1)
    samples = antenna.get_samples(8*1024*2*2)[0][0]

    num_taps = 8
    num_branches = 1024
    filterbank = stg.voltage.PolyphaseFilterbank(num_taps=num_taps,
                                                 num_branches=num_branches)
    X_f = filterbank.channelize(samples)

    X = stg.voltage.polyphase_filterbank.get_pfb_voltages(samples, 
                                                          num_taps=num_taps, 
                                                          num_branches=num_branches)
    
    assert_allclose(np.abs(X_f), np.abs(X)[:, :-1])


def test_vectorized_pfb_frontend_matches_reference():
    rng = np.random.default_rng(123)
    num_taps = 4
    num_branches = 32
    filterbank = stg.voltage.PolyphaseFilterbank(num_taps=num_taps,
                                                 num_branches=num_branches)
    samples = rng.standard_normal(num_taps * num_branches * 5)

    expected = pfb_frontend_reference(samples,
                                      filterbank.window,
                                      num_taps,
                                      num_branches)
    actual = pfb_frontend(samples,
                          filterbank.window,
                          num_taps,
                          num_branches,
                          method="vectorized")

    assert_allclose(actual, expected)


def test_pfb_frontend_auto_matches_reference():
    rng = np.random.default_rng(124)
    num_taps = 4
    num_branches = 32
    filterbank = stg.voltage.PolyphaseFilterbank(num_taps=num_taps,
                                                 num_branches=num_branches)
    samples = rng.standard_normal(num_taps * num_branches * 5)

    expected = pfb_frontend_reference(samples,
                                      filterbank.window,
                                      num_taps,
                                      num_branches)
    actual = pfb_frontend(samples,
                          filterbank.window,
                          num_taps,
                          num_branches,
                          method="auto")

    assert_allclose(actual, expected)


def test_selected_coarse_channelize_matches_full_slice():
    rng = np.random.default_rng(321)
    num_taps = 4
    num_branches = 64
    samples = rng.standard_normal(num_taps * num_branches * 6)

    full_filterbank = stg.voltage.PolyphaseFilterbank(num_taps=num_taps,
                                                      num_branches=num_branches)
    selected_filterbank = stg.voltage.PolyphaseFilterbank(num_taps=num_taps,
                                                          num_branches=num_branches)

    full = full_filterbank.channelize(samples,
                                      cache=False,
                                      method="full")
    selected = selected_filterbank.channelize(samples,
                                             cache=False,
                                             start_chan=3,
                                             num_chans=4,
                                             method="selected")

    assert_allclose(selected, full[:, 3:7], rtol=1e-12, atol=1e-12)


def test_selected_coarse_channelize_drifting_tone_matches_full_slice():
    num_taps = 4
    num_branches = 64
    num_samples = num_taps * num_branches * 6
    ts = np.arange(num_samples) / 3e9
    tone = np.cos(2 * np.pi * (1.5e8 * ts + 0.5 * 5e5 * ts**2))

    full_filterbank = stg.voltage.PolyphaseFilterbank(num_taps=num_taps,
                                                      num_branches=num_branches)
    selected_filterbank = stg.voltage.PolyphaseFilterbank(num_taps=num_taps,
                                                          num_branches=num_branches)

    full = full_filterbank.channelize(tone,
                                      cache=False,
                                      method="full")
    selected = selected_filterbank.channelize(tone,
                                             cache=False,
                                             start_chan=1,
                                             num_chans=2,
                                             method="selected")

    assert_allclose(selected, full[:, 1:3], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"method": "bad"}, "method must be"),
        ({"start_chan": -1}, "start_chan must be"),
        ({"start_chan": 0, "num_chans": 0}, "out of bounds"),
        ({"start_chan": 33, "num_chans": 1}, "out of bounds"),
    ],
)
def test_channelize_rejects_invalid_selection(kwargs, message):
    num_taps = 4
    num_branches = 64
    samples = np.ones(num_taps * num_branches * 4)
    filterbank = stg.voltage.PolyphaseFilterbank(num_taps=num_taps,
                                                 num_branches=num_branches)

    with pytest.raises(ValueError, match=message):
        filterbank.channelize(samples, cache=False, **kwargs)


def test_pfb_frontend_rejects_unknown_method():
    num_taps = 4
    num_branches = 32
    filterbank = stg.voltage.PolyphaseFilterbank(num_taps=num_taps,
                                                 num_branches=num_branches)
    samples = np.ones(num_taps * num_branches * 4)

    with pytest.raises(ValueError, match="method must be"):
        pfb_frontend(samples,
                     filterbank.window,
                     num_taps,
                     num_branches,
                     method="bad")
