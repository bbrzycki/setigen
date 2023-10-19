import pytest
import copy
import numpy as np
from numpy.testing import assert_allclose

from astropy import units as u
import setigen as stg


def test_get_distributions(tmp_path):
    dummy_frame = stg.Frame(shape=(16, 256*10), seed=0)
    dummy_frame.add_noise(1)
    dummy_fn = tmp_path / "big_frame.fil"
    dummy_frame.save_fil(dummy_fn)

    x_mean_array, x_std_array, x_min_array = stg.get_parameter_distributions(dummy_fn,
                                                                             fchans=256)
    assert len(x_mean_array) == len(x_std_array) == len(x_min_array) == 10

    x_mean_array = stg.get_mean_distribution(dummy_fn, fchans=256)
    assert len(x_mean_array) == 10

