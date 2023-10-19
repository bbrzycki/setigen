import pytest
from pathlib import Path
import copy
import numpy as np
from numpy.testing import assert_allclose

from astropy import units as u
import setigen as stg


@pytest.fixture()
def frame_setup():
    frame = stg.Frame(shape=(16, 256), seed=0)
    frame.add_noise(10)
    frame.add_constant_signal(frame.get_frequency(64),
                            0, 20, frame.df, f_profile_type="box")
    return frame


def test_sigma_clip_norm(frame_setup):
    frame = copy.deepcopy(frame_setup)
    n_frame = stg.sigma_clip_norm(frame)
    assert n_frame.get_noise_stats() == pytest.approx((-1.359109385231543e-15, 1.0))
    assert np.max(n_frame.data) == pytest.approx(22.575290615415213)
    n_frame.plot("px", db=False)

    frame = copy.deepcopy(frame_setup)
    n_frame = stg.sigma_clip_norm(frame, axis=0)
    assert n_frame.get_noise_stats() == pytest.approx((-6.600005856485898e-17, 1.0))
    assert np.max(n_frame.data) == pytest.approx(5.534988857620322)
    n_frame.plot("px", db=False)

    frame = copy.deepcopy(frame_setup)
    n_frame = stg.sigma_clip_norm(frame, axis=1)
    assert n_frame.get_noise_stats() == pytest.approx((-0.0007296582003336231, 0.9990398724412639))
    assert np.max(n_frame.data) == pytest.approx(25.27298782997925)
    n_frame.plot("px", db=False)


def test_sliding_norm(frame_setup):
    frame = copy.deepcopy(frame_setup)
    data = stg.sliding_norm(frame.data)
    assert np.mean(data) == pytest.approx(0)
    assert np.std(data) == pytest.approx(1)
    assert np.max(data) == pytest.approx(3.13907187881519)

    data = stg.sliding_norm(frame.data, 
                            cols=8, 
                            exclude=0.1, 
                            db=True, 
                            use_median=True)
    assert np.mean(data) == pytest.approx(0.2706269225103587)
    assert np.std(data) == pytest.approx(1.9841913710727954)
    assert np.max(data) == pytest.approx(19.331090277856557)


def test_blimpy_clip():
    assert_allclose(stg.blimpy_clip(np.arange(10), exclude=0.2),
                    np.array([7, 6, 5, 4, 3, 2, 1, 0]))


def test_max_norm(frame_setup):
    frame = copy.deepcopy(frame_setup)
    assert np.max(stg.max_norm(frame.data)) == pytest.approx(1)