import pytest
import copy
import numpy as np
from numpy.testing import assert_allclose

import os
from astropy import units as u
import setigen as stg


@pytest.fixture()
def frame_setup_no_data():
    frame = stg.Frame(fchans=1024*u.pixel,
                      tchans=32*u.pixel,
                      df=2.7939677238464355*u.Hz,
                      dt=18.25361108*u.s,
                      fch1=6095.214842353016*u.MHz)
    return frame


@pytest.fixture()
def constant_signal_data():
    my_path = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(my_path, 'assets/test_frame_data.npy')
    return np.load(path)


def test_setup_no_data(frame_setup_no_data):
    frame = copy.deepcopy(frame_setup_no_data)
    assert frame.fchans == 1024
    assert frame.tchans == 32
    assert frame.shape == (32, 1024)

    assert frame.df == pytest.approx(2.7939677238464355)
    assert frame.dt == pytest.approx(18.25361108)
    assert frame.fmax == pytest.approx(6095214842.353016)
    assert frame.fmin == pytest.approx(6095211981.330067)

    assert_allclose(frame.data, np.zeros((32, 1024)))
    assert frame.mean == frame.noise_mean == 0
    assert frame.std == frame.noise_std == 0
    assert frame.min == frame.noise_min == 0


def test_constant_signal_from_add_signal(frame_setup_no_data,
                                         constant_signal_data):
    frame = copy.deepcopy(frame_setup_no_data)
    signal = frame.add_signal(stg.constant_path(f_start=frame.fs[200],
                                                drift_rate=2*u.Hz/u.s),
                              stg.constant_t_profile(level=1),
                              stg.gaussian_f_profile(width=50*u.Hz),
                              stg.constant_bp_profile(level=1))
    assert_allclose(signal, frame.get_data(db=False))
    assert_allclose(frame.get_data(db=False), constant_signal_data)


def test_constant_signal_from_add_constant_signal(frame_setup_no_data,
                                                  constant_signal_data):
    frame = copy.deepcopy(frame_setup_no_data)
    signal = frame.add_constant_signal(f_start=frame.fs[200],
                                       drift_rate=2*u.Hz/u.s,
                                       level=1,
                                       width=50*u.Hz,
                                       f_profile_type='gaussian')
    assert_allclose(signal, frame.get_data(db=False))
    assert_allclose(frame.get_data(db=False), constant_signal_data, atol=1e-4)