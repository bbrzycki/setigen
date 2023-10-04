import pytest
import copy

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
                                    source_name=f"Obs{i}"))

    cad = stg.Cadence(frame_list=frame_list)
    return cad


def test_file_saves(cadence_setup, t_start_setup, tmp_path):
    cad = copy.deepcopy(cadence_setup)
    for i, frame in enumerate(cad):
        filename = tmp_path / f"{frame.source_name}.h5"
        frame.save_h5(filename=filename)

        frame = stg.Frame(filename)
        assert frame.t_start == pytest.approx(t_start_setup[i])