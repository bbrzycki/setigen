import pytest
from pathlib import Path
import copy
import numpy as np
from numpy.testing import assert_allclose

from astropy import units as u
import setigen as stg


def test_squared():
    assert_allclose(stg.squared_path(0, 1)(np.arange(8)),
                    np.array([0., 0.5, 2., 4.5, 8., 12.5, 18., 24.5]))


def test_sine_path():
    assert_allclose(stg.sine_path(0, 1, 4, 2)(np.arange(8)),
                    np.array([0., 3., 2., 1., 4., 7., 6., 5.]))


def test_rfi_path():
    assert_allclose(stg.simple_rfi_path(0, 
                                        1, 
                                        3, 
                                        spread_type="uniform", 
                                        rfi_type="stationary", 
                                        seed=0)(np.arange(8)),
                    [0.41088506, 0.30936014, 0.62292057, 1.54958291, 
                     4.93981072, 6.23826673, 6.31990733, 7.68848968])
    assert_allclose(stg.simple_rfi_path(0, 
                                        1, 
                                        3, 
                                        spread_type="uniform", 
                                        rfi_type="random_walk", 
                                        seed=0)(np.arange(8)),
                    [0.41088506, 0.7202452, 0.34316578, -0.10725132,
                     1.8325594, 4.07082613, 5.39073346, 7.07922314])
    assert_allclose(stg.simple_rfi_path(0, 
                                        1, 
                                        3, 
                                        spread_type="normal", 
                                        rfi_type="stationary", 
                                        seed=0)(np.arange(8)),
                    [0.16017813, 0.83170069, 2.81588738, 3.13364093, 
                     3.31756649, 5.46066584, 7.6612735, 8.20656476])
    assert_allclose(stg.simple_rfi_path(0, 
                                        1, 
                                        3, 
                                        spread_type="normal", 
                                        rfi_type="random_walk", 
                                        seed=0)(np.arange(8)),
                    [0.16017813, 0.99187882, 2.80776619, 3.94140713, 
                     4.25897361, 5.71963946, 8.38091296, 10.58747772])
