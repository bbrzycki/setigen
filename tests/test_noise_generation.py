import pytest
import copy
import numpy as np

from astropy import units as u
import setigen as stg
import blimpy as bl


@pytest.fixture()
def frame_setup_hres():
    frame = stg.Frame(fchans=1024*u.pixel,
                      tchans=32*u.pixel,
                      df=2.7939677238464355*u.Hz,
                      dt=18.253611008*u.s,
                      fch1=6095.214842353016*u.MHz)
    return frame


@pytest.fixture()
def frame_setup_very_hres():
    frame = stg.Frame(fchans=1024*u.pixel,
                      tchans=32*u.pixel,
                      df=1.3969838619232178*u.Hz,
                      dt=1.4316557653333333*u.s,
                      fch1=6095.214842353016*u.MHz)
    return frame


def test_hres_chi2(frame_setup_hres):
    frame = copy.deepcopy(frame_setup_hres)
    frame.add_noise(10, noise_type='chi2')
    
    assert abs(np.mean(frame.get_data()) / 10 - 1) < 0.02
    assert abs(np.std(frame.get_data()) / (10 * (2 / (4 * 51))**0.5) - 1) < 0.02
    
    
def test_very_hres_chi2(frame_setup_very_hres):
    frame = copy.deepcopy(frame_setup_very_hres)
    frame.add_noise(10, noise_type='chi2')
    
    assert abs(np.mean(frame.get_data()) / 10 - 1) < 0.02
    assert abs(np.std(frame.get_data()) / (10 * (2 / (4 * 2))**0.5) - 1) < 0.02