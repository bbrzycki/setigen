import importlib.util

import numpy as np
import pytest

import setigen as stg
from setigen.voltage import _array_backend


class _FakeCupyModule:
    __name__ = "cupy"

    @staticmethod
    def asarray(array):
        return np.asarray(array)

    @staticmethod
    def asnumpy(array):
        return np.asarray(array)


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


def test_auto_backend_env_falls_back_to_numpy_when_cupy_missing(monkeypatch):
    monkeypatch.setenv("SETIGEN_ENABLE_GPU", "1")
    monkeypatch.delitem(_array_backend._modules, "cupy", raising=False)
    monkeypatch.setattr(_array_backend.importlib, "import_module", lambda name: (_ for _ in ()).throw(ImportError()))

    stg.voltage.set_backend("auto")

    assert _array_backend.get_array_module("auto") is np
    assert stg.voltage.get_backend() == "numpy"


def test_array_backend_proxy_and_asnumpy_use_active_module(monkeypatch):
    fake_cupy = _FakeCupyModule()
    monkeypatch.setitem(_array_backend._modules, "cupy", fake_cupy)
    stg.voltage.set_backend("cupy")
    try:
        assert _array_backend.xp.__name__ == "cupy"
        assert _array_backend.xp.asarray([1]).shape == (1,)
        assert repr(_array_backend.xp) == repr(fake_cupy)
        assert np.array_equal(_array_backend.asnumpy([1, 2, 3]), np.array([1, 2, 3]))
    finally:
        stg.voltage.set_backend("numpy")


def test_array_backend_numpy_helpers(monkeypatch):
    monkeypatch.delenv("SETIGEN_ENABLE_GPU", raising=False)
    stg.voltage.set_backend("auto")

    assert _array_backend.get_array_module() is np
    assert _array_backend.get_array_module("numpy") is np
    assert np.array_equal(_array_backend.asnumpy([4, 5]), np.array([4, 5]))


def test_get_array_module_explicit_cupy_raises_when_missing(monkeypatch):
    monkeypatch.delitem(_array_backend._modules, "cupy", raising=False)
    monkeypatch.setattr(
        _array_backend.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(ImportError()),
    )

    with pytest.raises(ImportError, match="CuPy backend requested"):
        _array_backend.get_array_module("cupy")
