import pytest
import pickle

import numpy as np
from numpy.testing import assert_allclose
import setigen as stg
from astropy.time import Time
from pathlib import Path
import blimpy as bl


def test_frame_copy_mjd():
    frame = stg.Frame(shape=(16, 256), mjd=60000)
    assert frame.t_start == pytest.approx(1677283200.0)
    assert frame.waterfall is None

    frame.add_noise_from_obs()
    frame2 = frame.copy()
    assert frame.waterfall is None
    assert frame2.waterfall is None
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


def test_axis_semantics():
    frame = stg.Frame(tchans=3, fchans=4, dt=2, df=10, fch1=40)

    assert_allclose(frame.frequency_centers, frame.fs)
    assert_allclose(frame.frequency_edges, np.array([5, 15, 25, 35, 45]))
    assert_allclose(frame.time_starts, np.array([0, 2, 4]))
    assert_allclose(frame.time_centers, np.array([1, 3, 5]))
    assert_allclose(frame.time_edges, np.array([0, 2, 4, 6]))
    assert_allclose(frame.ts_ext, frame.time_edges)

    assert frame.get_drift_rate(0, 3) == pytest.approx(5)
    assert frame.get_drift_rate(0, 3, reference="centers") == pytest.approx(7.5)


def test_integrated_frames_preserve_observational_context():
    frame = stg.Frame(shape=(4, 8),
                      t_start=123456,
                      source_name="Target",
                      data=np.ones((4, 8)))
    frame.header = {"rawdatafile": "original.raw", "source_name": "Target"}
    frame.add_metadata({"drift_rate": 1.25})

    spectrum = stg.spectrum(frame, mode="sum")
    timeseries = stg.timeseries(frame, mode="sum")

    for derived in [spectrum, timeseries]:
        assert derived.t_start == frame.t_start
        assert derived.source_name == frame.source_name
        assert derived.header == frame.header
        assert derived.header is not frame.header
        assert derived.metadata["drift_rate"] == frame.metadata["drift_rate"]
        assert derived.metadata["fchans"] == derived.fchans
        assert derived.metadata["tchans"] == derived.tchans


def test_h5_ingestion_does_not_retain_live_waterfall(tmp_path):
    frame = stg.Frame(shape=(4, 8), seed=0)
    h5_path = tmp_path / "frame.h5"
    frame.save_hdf5(h5_path)

    loaded = stg.Frame(waterfall=h5_path)
    assert loaded.waterfall is None
    assert loaded.header is not None

    loaded.copy()
    pickle.dumps(loaded)

    adapter = loaded.get_waterfall()
    assert not hasattr(adapter.container, "h5")
    assert loaded.check_waterfall() is adapter


def test_waterfall_object_ingestion_does_not_mutate_source_h5(tmp_path):
    frame = stg.Frame(shape=(4, 8), seed=0)
    h5_path = tmp_path / "frame.h5"
    frame.save_hdf5(h5_path)

    waterfall = bl.Waterfall(str(h5_path))
    try:
        assert hasattr(waterfall.container, "h5")
        loaded = stg.Frame(waterfall=waterfall)
        assert loaded.waterfall is None
        assert hasattr(waterfall.container, "h5")

        sliced = loaded.get_slice(2, 6)
        assert sliced.shape == (4, 4)
        assert hasattr(waterfall.container, "h5")
    finally:
        waterfall.container.h5.close()
