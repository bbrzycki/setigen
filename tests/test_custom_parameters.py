import pytest
from pathlib import Path
import copy
import numpy as np
from numpy.testing import assert_allclose

from astropy import units as u
import setigen as stg
import blimpy as bl


@pytest.fixture()
def frame_setup_no_data():
    frame = stg.Frame(fchans=1024*u.pixel,
                      tchans=32*u.pixel,
                      df=2.7939677238464355*u.Hz,
                      dt=18.253611008*u.s,
                      fch1=6095.214842353016*u.MHz)
    return frame


@pytest.fixture()
def frame_setup_no_data_ascending():
    frame = stg.Frame(fchans=1024*u.pixel,
                      tchans=32*u.pixel,
                      df=2.7939677238464355*u.Hz,
                      dt=18.253611008*u.s,
                      fch1=6095211984.124035,
                      ascending=True)
    return frame


@pytest.fixture()
def constant_signal_data():
    # my_path = os.path.abspath(os.path.dirname(__file__))
    # path = os.path.join(my_path, 'assets/test_frame_data.npy')
    path = Path(__file__).resolve().parent / "assets/test_frame_data.npy"
    return np.load(path)


def test_setup_no_data(frame_setup_no_data):
    frame = copy.deepcopy(frame_setup_no_data)
    assert frame.fchans == 1024
    assert frame.tchans == 32
    assert frame.shape == (32, 1024)

    assert frame.df == pytest.approx(2.7939677238464355)
    assert frame.dt == pytest.approx(18.253611008)
    assert (frame.fmax - 6095214842.353016) == pytest.approx(0)
    assert (frame.fmin - 6095211984.124035) == pytest.approx(0)

    assert_allclose(frame.data, np.zeros((32, 1024)))
    assert frame.mean == frame.noise_mean == 0
    assert frame.std == frame.noise_std == 0
    
    
def test_setup_no_data_ascending(frame_setup_no_data_ascending):
    frame = copy.deepcopy(frame_setup_no_data_ascending)
    assert frame.fchans == 1024
    assert frame.tchans == 32
    assert frame.shape == (32, 1024)

    assert frame.df == pytest.approx(2.7939677238464355)
    assert frame.dt == pytest.approx(18.253611008)
    assert (frame.fmax - 6095214842.353016) == pytest.approx(0)
    assert (frame.fmin - 6095211984.124035) == pytest.approx(0)

    assert_allclose(frame.data, np.zeros((32, 1024)))
    assert frame.mean == frame.noise_mean == 0
    assert frame.std == frame.noise_std == 0
    
    
def test_index_calc(frame_setup_no_data, frame_setup_no_data_ascending):
    frame = copy.deepcopy(frame_setup_no_data)
    frame1 = copy.deepcopy(frame_setup_no_data_ascending)
    assert_allclose(frame.fs, frame1.fs)
    assert frame.get_index(frame.fch1) == 1023
    assert frame1.get_index(frame1.fch1) == 0
    assert frame.get_index(6095213000) == frame1.get_index(6095213000)
    assert frame.get_index(42) == frame1.get_index(42)
    assert frame.get_frequency(200) == pytest.approx(6095212542.91758)
    assert frame1.get_frequency(200) == pytest.approx(6095212542.91758)

    
def test_fil_io(frame_setup_no_data, tmp_path):
    frame = copy.deepcopy(frame_setup_no_data)

    fil_fn = tmp_path / 'temp.fil'
    frame.save_fil(fil_fn)

    temp_frame = stg.Frame(waterfall=fil_fn)
    assert_allclose(temp_frame.get_data(), frame.get_data())

    wf = bl.Waterfall(fil_fn)
    temp_frame = stg.Frame(waterfall=wf)
    assert_allclose(temp_frame.get_data(), frame.get_data())


def test_h5_io(frame_setup_no_data, tmp_path):
    frame = copy.deepcopy(frame_setup_no_data)

    fil_fn = tmp_path / 'temp.h5'
    frame.save_hdf5(fil_fn)

    temp_frame = stg.Frame(waterfall=fil_fn)
    assert_allclose(temp_frame.get_data(), frame.get_data())

    wf = bl.Waterfall(fil_fn)
    temp_frame = stg.Frame(waterfall=wf)
    assert_allclose(temp_frame.get_data(), frame.get_data())


def test_constant_signal_from_add_signal(frame_setup_no_data,
                                         constant_signal_data):
    frame = copy.deepcopy(frame_setup_no_data)
    signal = frame.add_signal(stg.constant_path(f_start=frame.get_frequency(200),
                                                drift_rate=2*u.Hz/u.s),
                              stg.constant_t_profile(level=1),
                              stg.gaussian_f_profile(width=50*u.Hz),
                              stg.constant_bp_profile(level=1))
    assert_allclose(signal, frame.get_data(db=False))
    assert_allclose(frame.get_data(db=False), constant_signal_data)


def test_constant_signal_from_add_constant_signal(frame_setup_no_data,
                                                  constant_signal_data):
    frame = copy.deepcopy(frame_setup_no_data)
    signal = frame.add_constant_signal(f_start=frame.get_frequency(200),
                                       drift_rate=2*u.Hz/u.s,
                                       level=1,
                                       width=50*u.Hz,
                                       f_profile_type='gaussian')
    assert_allclose(signal, frame.get_data(db=False))
    assert_allclose(frame.get_data(db=False), constant_signal_data, atol=1e-4)


def test_signal_from_arrays():
    frame = stg.Frame(tchans=3, fchans=3, dt=1, df=1, fch1=3)
    frame.add_signal(path=[2, 1, 3],
                     t_profile=[1, 0.5, 1],
                     f_profile=stg.box_f_profile(width=1),
                     bp_profile=[1, 0.5, 1])

    data = np.array([[0., 0.5, 0.], [0.5, 0., 0.], [0., 0., 1.]])
    assert_allclose(data, frame.get_data(db=False))


def test_signal_from_floats():
    frame = stg.Frame(tchans=3, fchans=3, dt=1, df=1, fch1=3)
    frame.add_signal(path=3,
                     t_profile=2,
                     f_profile=stg.box_f_profile(width=1),
                     bp_profile=3)

    data = np.array([[0., 0., 6.], [0., 0., 6.], [0., 0., 6.]])
    assert_allclose(data, frame.get_data(db=False))
