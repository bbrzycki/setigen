import pytest

import numpy as np
from numpy.testing import assert_allclose
import setigen as stg
from astropy.time import Time
from pathlib import Path


def test_frame_copy_mjd():
    frame = stg.Frame(shape=(16, 256), mjd=60000)
    assert frame.t_start == pytest.approx(1677283200.0)

    frame.add_noise_from_obs()
    frame2 = frame.copy()
    assert_allclose(frame.data, frame2.data)


def test_from_data():
    df = 5
    dt = 20
    fch1 = 4000
    fchans = 256
    tchans = 32
    data = np.ones(shape=(tchans, fchans))

    frame = stg.Frame.from_data(df=df, 
                                dt=dt,
                                fch1=fch1,
                                ascending=True,
                                data=data,
                                metadata={"test_key": "test_val"})
    
    assert frame.metadata["test_key"] == "test_val"
    assert_allclose(frame.data, np.ones(shape=(tchans, fchans)))
    assert frame.df == df 
    assert frame.dt == dt
    assert frame.fchans == fchans 
    assert frame.tchans == tchans 


def test_from_backend_params():
    data = np.ones((128, 256))
    frame = stg.Frame.from_backend_params(fchans=256, 
                                          obs_length=600, 
                                          sample_rate=3e9, 
                                          num_branches=1024, 
                                          fftlength=524288,
                                          int_factor=26,
                                          fch1=6e9,
                                          ascending=False,
                                          data=data)
    
    assert_allclose(frame.data, data)
    assert frame.df == pytest.approx(5.587935447692871)
    assert frame.dt == pytest.approx(4.652881237333333)
    assert frame.fchans == 256 
    assert frame.tchans == 128 


def test_init_precedence_rules():
    data = np.ones((3, 4))
    frame = stg.Frame(fchans=99,
                      tchans=88,
                      shape=(3, 4),
                      data=data,
                      mjd=60000,
                      t_start=12345,
                      source_name="Test source")

    assert frame.shape == (3, 4)
    assert frame.fchans == 4
    assert frame.tchans == 3
    assert frame.t_start == pytest.approx(Time(60000, format='mjd').unix)
    assert frame.source_name == "Test source"
    assert_allclose(frame.data, data)


def test_data_shape_takes_precedence_over_explicit_dimensions():
    data = np.ones((2, 5))
    frame = stg.Frame(fchans=10,
                      tchans=20,
                      data=data)

    assert frame.shape == (2, 5)
    assert frame.fchans == 5
    assert frame.tchans == 2
    assert_allclose(frame.data, data)


def test_synthetic_mode_takes_precedence_over_waterfall():
    path = Path(__file__).resolve().parent / "assets/sample.fil"
    frame = stg.Frame(waterfall=path,
                      shape=(2, 3))

    assert frame.shape == (2, 3)
    assert frame.waterfall is None


def test_waterfall_path_selection_kwargs():
    path = Path(__file__).resolve().parent / "assets/sample.fil"
    full_frame = stg.Frame(waterfall=path)

    subset_start = full_frame.get_frequency(100) * 1e-6
    subset_stop = full_frame.get_frequency(900) * 1e-6
    subset_frame = stg.Frame(waterfall=path,
                             f_start=subset_start,
                             f_stop=subset_stop)

    assert subset_frame.fchans < full_frame.fchans
