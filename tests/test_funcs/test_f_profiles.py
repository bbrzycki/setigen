import pytest
from pathlib import Path
import copy
import numpy as np
from numpy.testing import assert_allclose

from astropy import units as u
import setigen as stg


def test_sinc2_width_mode_enum_matches_string():
    frequencies = np.linspace(-5, 5, 11)
    center = 0

    string_profile = stg.sinc2_f_profile(width=4,
                                         width_mode="fwhm",
                                         trunc=False)
    enum_profile = stg.sinc2_f_profile(width=4,
                                       width_mode=stg.WidthMode.FWHM,
                                       trunc=False)

    assert_allclose(string_profile(frequencies, center),
                    enum_profile(frequencies, center))

