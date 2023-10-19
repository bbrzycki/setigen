import pytest
from pathlib import Path
import copy
import numpy as np
from numpy.testing import assert_allclose

from astropy import units as u
import setigen as stg


def test_split_waterfall(tmp_path):
    dummy_frame = stg.Frame(shape=(16, 256*10))
    dummy_fn = tmp_path / "big_frame.fil"
    dummy_frame.save_fil(dummy_fn)

    fns = stg.split_fil(dummy_fn, tmp_path, fchans=256)
    for fn in fns:
        assert stg.Frame(fn).shape == (16, 256)


def test_split_array():
    dummy_array = np.empty(shape=(16, 256*10))
    split_array = stg.split_array(dummy_array)
    assert split_array.shape == (1, 16, 256*10)
    split_array = stg.split_array(dummy_array, 
                                  f_sample_num=256, 
                                  t_sample_num=16)
    assert split_array.shape == (10, 16, 256)

    dummy_array = np.empty(shape=(16+2, 256*10+34))
    split_array = stg.split_array(dummy_array)
    assert split_array.shape == (1, 16+2, 256*10+34)
    split_array = stg.split_array(dummy_array, 
                                  f_sample_num=256, 
                                  t_sample_num=16, 
                                  f_trim=True, 
                                  t_trim=True)
    assert split_array.shape == (10, 16, 256)