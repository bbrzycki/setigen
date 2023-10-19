import pytest
import copy
import numpy as np
from numpy.testing import assert_allclose

from astropy import units as u
import setigen as stg


def test_utils(tmp_path):
    frame = stg.Frame(shape=(16, 256), seed=0)
    frame.add_noise(1)
    fil_path = tmp_path / "tmp.fil"
    frame.save_fil(fil_path)

    assert stg.max_freq(fil_path) == pytest.approx(frame.fmax / 1e6)
    assert stg.min_freq(fil_path) == pytest.approx(frame.fmin / 1e6)
    assert_allclose(stg.get_data(fil_path)[:, ::-1], frame.get_data())
    assert_allclose(stg.get_data(fil_path, db=True)[:, ::-1], frame.get_data(db=True), rtol=1e-3)
    assert_allclose(np.sort(stg.get_fs(fil_path)), frame.fs / 1e6)
    assert_allclose(stg.get_ts(fil_path), frame.ts)