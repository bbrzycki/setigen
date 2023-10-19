import pytest
from pathlib import Path
import copy
import numpy as np
from numpy.testing import assert_allclose

from astropy import units as u
import setigen as stg


def test_constant_t_profile():
    assert stg.constant_t_profile(level=1)(0) == 1


def test_sine_t_profile():
    assert_allclose(stg.sine_t_profile(4, 1, 1, 1)(np.arange(8)),
                    np.array([2., 1., 0., 1., 2., 1., 0., 1.]))


def test_periodic_gaussian():
    assert_allclose(stg.periodic_gaussian_t_profile(2, 4, 1, 2, 
                                                    pulse_direction="rand", 
                                                    pnum=3, 
                                                    amplitude=1, 
                                                    level=1, 
                                                    min_level=0, 
                                                    seed=0)(np.arange(8)),
                    np.array([0.00865991, 0.57557674, 0.94333846, 0.80716901, 
                              0.18539281, 0.13292513, 0.72142534, 0.54523888]))
    assert_allclose(stg.periodic_gaussian_t_profile(2, 4, 1, 2, 
                                                    pulse_direction="up", 
                                                    pnum=4, 
                                                    amplitude=1, 
                                                    level=1, 
                                                    min_level=0, 
                                                    seed=0)(np.arange(8)),
                    np.array([1.99134009, 1.42442326, 1.05666154, 1.19283099, 
                              1.81460719, 1.86707487, 1.27857466, 1.45476112]))
    assert_allclose(stg.periodic_gaussian_t_profile(2, 4, 1, 2, 
                                                    pulse_direction="down", 
                                                    pnum=4, 
                                                    amplitude=1, 
                                                    level=1, 
                                                    min_level=0, 
                                                    seed=0)(np.arange(8)),
                    np.array([0.00865991, 0.57557674, 0.94333846, 0.80716901, 
                              0.18539281, 0.13292513, 0.72142534, 0.54523888]))