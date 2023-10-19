import pytest
import copy
import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_allclose

from astropy import units as u
import setigen as stg
from astropy.time import Time


@pytest.fixture()
def t_start_setup():
    mjd_start = 60000
    obs_length = 300
    slew_time = 15

    t_start_arr = [Time(mjd_start, format='mjd').unix]
    for i in range(1, 6):
        t_start_arr.append(t_start_arr[i - 1] + obs_length + slew_time)
    return t_start_arr


@pytest.fixture()
def cadence_setup(t_start_setup):
    frame_list = []
    for i in range(6):
        frame_list.append(stg.Frame(fchans=256, 
                                    tchans=16, 
                                    t_start=t_start_setup[i], 
                                    source_name=f"Obs{i}",
                                    seed=i))
        
    cad = stg.Cadence(frame_list=frame_list)
    cad.apply(lambda f: f.add_noise(1))
    return cad


@pytest.fixture()
def frame_setup():
    frame = stg.Frame(fchans=1024,
                      tchans=32,
                      df=2.7939677238464355,
                      dt=18.253611008,
                      fch1=6095.214842353016*u.MHz,
                      seed=0)
    frame.add_noise(1)
    return frame


def test_plot_extents():
    frame = stg.Frame(fchans=3e3,
                      tchans=16,
                      df=1e6,
                      dt=18.253611008)
    assert stg.plots._get_extent_units(frame) == (1e9, "GHz")
    frame = stg.Frame(fchans=3e3,
                      tchans=16,
                      df=1e3,
                      dt=18.253611008)
    assert stg.plots._get_extent_units(frame) == (1e6, "MHz")
    frame = stg.Frame(fchans=1024,
                      tchans=16,
                      df=3,
                      dt=18.253611008)
    assert stg.plots._get_extent_units(frame) == (1e3, "kHz")


def test_plot_frame_options(frame_setup):
    frame = copy.deepcopy(frame_setup)
    frame.plot(xtype="fmid")
    plt.show()
    frame.plot(xtype="fmin", colorbar=False)
    plt.show()
    frame.plot(xtype="f", minor_ticks=True, label=True)
    plt.show()
    frame.plot(xtype="px", db=False, grid=True)
    plt.show()


def test_plot_cadence_options(cadence_setup):
    cad = copy.deepcopy(cadence_setup)
    cad.plot(xtype="fmid", slew_times=True)
    plt.show()
    cad.plot(xtype="fmin", colorbar=False, title=True)
    plt.show()
    cad.plot(xtype="f", minor_ticks=True, labels=False)
    plt.show()
    cad.plot(xtype="px", db=False, grid=True)
    plt.show()
