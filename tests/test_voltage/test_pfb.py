import pytest
import copy
import numpy as np
from numpy.testing import assert_allclose

from astropy import units as u
import setigen as stg


def test_filterbank():
    filterbank = stg.voltage.PolyphaseFilterbank(num_taps=8,
                                                 num_branches=1024)
    response = filterbank.get_response(fftlength=512)
    assert np.max(response) / np.mean(response) == pytest.approx(1.1101855941416805)
    assert len(response) == 256

    assert len(filterbank.tile_response(num_chans=64, fftlength=512)) == 64 * 512


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

    