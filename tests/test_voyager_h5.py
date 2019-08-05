import pytest
import copy

import os
from astropy import units as u
import setigen as stg


@pytest.fixture()
def frame_setup_from_h5():
    my_path = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(my_path, 'assets/Voyager1.single_coarse.fine_res.h5')
    frame = stg.Frame(fil=path)
    return frame


def test_setup_from_h5(frame_setup_from_h5):
    frame = frame_setup_from_h5
    assert frame.fchans == 1048576
    assert frame.tchans == 16
    assert frame.shape == (16, 1048576)

    assert frame.df == pytest.approx(2.7939677238464355)
    assert frame.dt == pytest.approx(18.25361108)
    assert frame.fmax == pytest.approx(8421386717.353016)
    assert frame.fmin == pytest.approx(8418457029.853016)

    assert frame.mean == pytest.approx(10394075000.0)
    # assert frame.noise_mean == pytest.approx(10394075000.0)
    assert frame.std == pytest.approx(54797877000.0)
    # assert frame.noise_std == pytest.approx(10394075000.0)
    assert frame.min == pytest.approx(3200441300.0)
    # assert frame.noise_min == pytest.approx(10394075000.0)
