import pytest
from pathlib import Path
import copy
import numpy as np
from numpy.testing import assert_allclose

from astropy import units as u
import setigen as stg


def test_dedrift():
    for ascending in [False, True]:
        for sign in [1, -1]:
            frame = stg.Frame(shape=(16, 512), ascending=ascending, seed=0)   
            drift_rate = sign * frame.unit_drift_rate 
            frame.add_constant_signal(frame.get_frequency(64),
                                      drift_rate=drift_rate,
                                      level=1,
                                      width=frame.df,
                                      f_profile_type="box")
            frame.add_metadata({"drift_rate": drift_rate})

            dd_frame = stg.dedrift(frame)
            assert np.max(dd_frame.integrate()) == pytest.approx(1)

            dd_frame = stg.dedrift(frame, drift_rate=drift_rate)
            assert np.max(dd_frame.integrate()) == pytest.approx(1)