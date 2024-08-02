import pytest
from pathlib import Path
import copy
import numpy as np
from numpy.testing import assert_allclose

from astropy import units as u
import setigen as stg


def test_noise():
    frame = stg.Frame(shape=(16, 256), seed=0)
    frame.add_noise(0, 1, noise_type="gaussian")
    assert frame.get_noise_stats() == (0, 1)

    frame.add_noise(0, 1, -1, noise_type="gaussian")
    assert frame.get_total_stats() == pytest.approx((0.0856390392253974, 
                                                     1.3226016826044575))

    frame.zero_data()
    assert frame.get_noise_stats() == (0, 0)

    frame.add_noise_from_obs(x_mean_array=[3,4,5], 
                             x_std_array=[1,2,3], 
                             share_index=True,
                             noise_type="gaussian")
    assert frame.get_noise_stats() == (5, 3)

    frame.zero_data()
    frame.add_noise_from_obs(x_mean_array=[3,4,5], 
                             x_std_array=[1,2,3], 
                             share_index=False,
                             noise_type="gaussian")
    assert frame.get_noise_stats() == (4, 3)

    frame.zero_data()
    frame.add_noise_from_obs(x_mean_array=[3,4,5], 
                             x_std_array=[1,2,3], 
                             x_min_array=[1,0,0],
                             share_index=True,
                             noise_type="gaussian")
    assert frame.get_noise_stats() == (3, 1)

    frame.zero_data()
    frame.add_noise_from_obs(x_mean_array=[3,4,5], 
                             x_std_array=[1,2,3], 
                             x_min_array=[1,0,0],
                             share_index=False,
                             noise_type="gaussian")
    assert frame.get_noise_stats() == (5, 2)

    frame.add_noise_from_obs(x_mean_array=[3,4,5], 
                             x_std_array=[1,2,3], 
                             x_min_array=[1,0,0],
                             share_index=False,
                             noise_type="gaussian")
    assert frame.get_noise_stats() == pytest.approx((9.075061618621975, 
                                                     2.732297391771127))


def test_injection_options():
    frame = stg.Frame(shape=(16, 256), seed=0)
    frame.add_noise(4e6)

    frame.add_signal(stg.constant_path(f_start=frame.get_frequency(frame.fchans//2),
                                       drift_rate=1*u.Hz/u.s),
                     stg.constant_t_profile(level=frame.get_intensity(snr=30)),
                     stg.gaussian_f_profile(width=40*u.Hz),
                     integrate_path=True,
                     integrate_t_profile=True,
                     integrate_f_profile=True,
                     t_subsamples=10,
                     f_subsamples=10)
    assert np.max(frame.data) == pytest.approx(8041210.038828725)

    # Test constant signals
    frame = stg.Frame(shape=(16, 256), seed=0)
    frame.add_constant_signal(frame.get_frequency(frame.fchans//2),
                              drift_rate=0,
                              level=1,
                              width=2*frame.df,
                              f_profile_type="sinc2")
    assert np.max(stg.integrate(frame, mode='s')) == frame.tchans

    frame.zero_data()
    frame.add_constant_signal(frame.get_frequency(frame.fchans//2),
                              drift_rate=0,
                              level=1,
                              width=2*frame.df,
                              f_profile_type="gaussian")
    assert np.max(stg.integrate(frame, mode='s')) == frame.tchans

    frame.zero_data()
    frame.add_constant_signal(frame.get_frequency(frame.fchans//2),
                              drift_rate=0,
                              level=1,
                              width=2*frame.df,
                              f_profile_type="lorentzian")
    assert np.max(stg.integrate(frame, mode='s')) == frame.tchans

    frame.zero_data()
    frame.add_constant_signal(frame.get_frequency(frame.fchans//2),
                              drift_rate=0,
                              level=1,
                              width=2*frame.df,
                              f_profile_type="voigt")
    assert np.max(stg.integrate(frame, mode='s')) == frame.tchans

    frame.zero_data()
    frame.add_constant_signal(frame.get_frequency(frame.fchans//2),
                              drift_rate=0,
                              level=1,
                              width=2*frame.df,
                              f_profile_type="box")
    assert np.max(stg.integrate(frame, mode='s')) == frame.tchans
    

def test_injection_tools(tmp_path):
    frame = stg.Frame(shape=(16, 256), seed=0)
    assert frame.check_waterfall() is None 

    with pytest.raises(ValueError) as exc_info:
        intensity = frame.get_intensity(snr=100)
    assert exc_info.type is ValueError

    with pytest.raises(ValueError) as exc_info:
        snr = frame.get_snr(intensity=100)
    assert exc_info.type is ValueError

    frame.add_noise(1)
    assert frame.get_snr(intensity=100) == pytest.approx(4039.801975344831)
    assert isinstance(frame.get_info(), dict)

    frame.add_constant_signal(frame.get_frequency(frame.fchans//2),
                              drift_rate=1,
                              level=10,
                              width=2*frame.df,
                              f_profile_type="sinc2",
                              doppler_smearing=True)

    assert frame.get_drift_rate(0, frame.fchans//2) == pytest.approx(1.224510688924805)
    assert np.max(frame.get_data(db=True)) == pytest.approx(stg.db(np.max(frame.data)))

    frame.update_metadata({"new_key": "new_val"})
    assert frame.get_metadata()["new_key"] == "new_val"

    assert frame.get_slice(64, 192).shape == (16, 128)

    wf = frame.get_waterfall()
    assert frame.check_waterfall() == wf

    # Check numpy save
    npy_path = tmp_path / 'data.npy'
    frame.save_npy(npy_path)
    data = copy.deepcopy(frame.data)
    frame.load_npy(npy_path)
    assert_allclose(data, frame.data)

    # Check pickle save
    pickle_path = tmp_path / 'frame.pickle'
    frame.save_pickle(pickle_path)
    new_frame = stg.Frame.load_pickle(pickle_path)
    assert_allclose(frame.data, new_frame.data)
    assert frame.metadata == new_frame.metadata