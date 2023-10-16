import pytest

import numpy as np
from numpy.testing import assert_allclose
import setigen as stg
from astropy.time import Time


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

