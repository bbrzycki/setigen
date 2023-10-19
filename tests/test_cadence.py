import pytest
import copy
import numpy as np
from numpy.testing import assert_allclose

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
    return cad


def test_file_saves(cadence_setup, t_start_setup, tmp_path):
    cad = copy.deepcopy(cadence_setup)
    for i, frame in enumerate(cad):
        filename = tmp_path / f"{frame.source_name}.h5"
        frame.save_h5(filename=filename)

        frame = stg.Frame(filename)
        assert frame.t_start == pytest.approx(t_start_setup[i])


def test_list_operations(cadence_setup):
    cad = copy.deepcopy(cadence_setup)
    frame = cad.pop()
    assert len(cad) == 5 
    cad[0] = stg.Frame(fchans=256, 
                       tchans=16, 
                       t_start=Time(60000, format='mjd').unix,
                       source_name="New obs")
    assert len(cad) == 5
    assert cad[0].source_name == "New obs"

    cad.append(stg.Frame(fchans=256, 
                         tchans=16, 
                         t_start=frame.t_start))
    assert len(cad) == 6 
    assert len(cad[0::2]) == 3
    assert len(cad[[1,4]]) == 2




def test_empty_attributes():
    cad = stg.Cadence()
    assert cad.t_start is None
    assert cad.fch1 is None
    assert cad.ascending is None
    assert cad.fmin is None
    assert cad.fmax is None
    assert cad.fmid is None
    assert cad.df is None
    assert cad.dt is None
    assert cad.fchans is None
    assert cad.tchans is None
    assert cad.obs_range is None
    assert cad.consolidate() is None


def test_cadence_attributes(cadence_setup):
    cad = copy.deepcopy(cadence_setup)
    assert cad.t_start == cad[0].t_start
    assert cad.fch1 == cad[0].fch1
    assert cad.ascending == cad[0].ascending
    assert cad.fmin == cad[0].fmin
    assert cad.fmax == cad[0].fmax
    assert cad.fmid == cad[0].fmid
    assert cad.df == cad[0].df
    assert cad.dt == cad[0].dt
    assert cad.fchans == cad[0].fchans
    assert cad.tchans == 6 * cad[0].tchans
    assert cad.obs_range == cad[-1].t_stop - cad[0].t_start
    assert "Frame" in str(cad)
    assert cad.consolidate().shape == (cad.tchans, cad.fchans)


def test_slew_times():
    frame_list = []
    for i in range(6):
        frame_list.append(stg.Frame(fchans=256, 
                                    tchans=16, 
                                    source_name=f"Obs{i}"))
        
    cad = stg.Cadence(frame_list=frame_list,
                      t_slew=30,
                      t_overwrite=True)
    assert_allclose(cad.slew_times, np.full(5, 30))


def test_ordered_cadence():
    mjd_start = 60000
    obs_length = 300
    slew_time = 15

    t_start_arr = [Time(mjd_start, format='mjd').unix]
    for i in range(1, 6):
        t_start_arr.append(t_start_arr[i - 1] + obs_length + slew_time)

    frame_list = []
    for i in range(6):
        frame_list.append(stg.Frame(fchans=256, 
                                    tchans=16, 
                                    t_start=t_start_arr[i], 
                                    source_name=f"Obs{i}"))

    cad = stg.OrderedCadence(frame_list=frame_list, order="ABACAD")
    cad[0] = stg.Frame(fchans=256, 
                       tchans=16, 
                       t_start=t_start_arr[0],
                       source_name="New obs")
    assert cad[0].source_name == "New obs"
    assert len(cad.by_label("A")) == 3
    assert len(cad.by_label("B")) == 1
    cad.pop()
    cad.set_order(order="ABACAB")
    cad.append(stg.Frame(fchans=256, 
                         tchans=16, 
                         t_start=t_start_arr[-1]))
    assert len(cad.by_label("B")) == 2


def test_basic_cadence_injection(cadence_setup):
    cad = copy.deepcopy(cadence_setup)
    cad.apply(lambda fr: fr.add_noise(1))
    cad[0::2].add_signal(stg.constant_path(f_start=cad[0].get_frequency(index=128),
                                           drift_rate=0.1),
                    stg.constant_t_profile(level=cad[0].get_intensity(snr=100)),
                    stg.sinc2_f_profile(width=2*cad[0].df, width_mode="crossing", trunc=False),
                    stg.constant_bp_profile(level=1),
                    doppler_smearing=True)

    for i in range(0, 6, 2):
        assert np.max(cad[i].integrate()) > 1.25
    for i in range(1, 6, 2):
        assert np.max(cad[i].integrate()) < 1.11
