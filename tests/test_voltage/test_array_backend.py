import importlib.util

import numpy as np
import pytest

import setigen as stg


def test_set_backend_numpy_forces_numpy():
    stg.voltage.set_backend("numpy")

    stream = stg.voltage.DataStream(seed=0)
    samples = stream.get_samples(8)

    assert stg.voltage.get_backend() == "numpy"
    assert isinstance(samples, np.ndarray)


def test_set_backend_rejects_unknown_backend():
    with pytest.raises(ValueError, match="backend must be"):
        stg.voltage.set_backend("not-a-backend")


def test_set_backend_cupy_validates_import():
    try:
        if importlib.util.find_spec("cupy") is None:
            with pytest.raises(ImportError, match="CuPy backend requested"):
                stg.voltage.set_backend("cupy")
        else:
            stg.voltage.set_backend("cupy")
            assert stg.voltage.get_backend() == "cupy"
    finally:
        stg.voltage.set_backend("numpy")
