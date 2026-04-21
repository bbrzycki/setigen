import pytest
import setigen as stg


def test_voigt():
    assert stg.funcs.func_utils.voigt_fwhm(10, 10) == pytest.approx(16.375959202100432)

    x = 10
    x0 = 2
    assert stg.funcs.func_utils.voigt(x, x0, sigma=5, gamma=0) == pytest.approx(stg.funcs.func_utils.gaussian(x, x0, sigma=5))
    assert stg.funcs.func_utils.voigt(x, x0, sigma=0, gamma=5) == pytest.approx(stg.funcs.func_utils.lorentzian(x, x0, gamma=5))
