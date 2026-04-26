from __future__ import annotations

import importlib
import os
from typing import Any, Literal

import numpy as np


ArrayBackend = Literal["auto", "numpy", "cupy"]

_backend: ArrayBackend = "auto"
_modules: dict[str, Any] = {"numpy": np}


def _env_backend() -> ArrayBackend:
    """Resolve the legacy environment-selected backend.

    Returns:
        ``"cupy"`` when ``SETIGEN_ENABLE_GPU=1``; otherwise ``"numpy"``.
    """
    return "cupy" if os.getenv("SETIGEN_ENABLE_GPU", "0") == "1" else "numpy"


def _validate_backend(backend: str) -> ArrayBackend:
    """Validate and normalize an array backend name.

    Args:
        backend: Backend name to validate.

    Returns:
        Validated backend name.

    Raises:
        ValueError: If the backend name is unsupported.
    """
    if backend not in ("auto", "numpy", "cupy"):
        raise ValueError("backend must be one of 'auto', 'numpy', or 'cupy'.")
    return backend  # type: ignore[return-value]


def _import_cupy() -> Any:
    """Import and cache CuPy.

    Returns:
        Imported CuPy module.

    Raises:
        ImportError: If CuPy is unavailable.
    """
    if "cupy" not in _modules:
        try:
            _modules["cupy"] = importlib.import_module("cupy")
        except ImportError as exc:
            raise ImportError(
                "CuPy backend requested, but CuPy is not installed. "
                "Install the CuPy package matching your CUDA runtime, or use "
                "stg.voltage.set_backend('numpy')."
            ) from exc
    return _modules["cupy"]


def set_backend(backend: str) -> None:
    """Set the default array backend for voltage synthesis.

    Call this before constructing voltage objects so their arrays, cached
    windows, and random generators are created with the intended backend.

    Args:
        backend: ``"numpy"``, ``"cupy"``, or ``"auto"``. ``"auto"`` follows
            the legacy ``SETIGEN_ENABLE_GPU`` environment variable policy.

    Raises:
        ImportError: If ``backend="cupy"`` is requested but CuPy is unavailable.
        ValueError: If the backend name is unsupported.
    """
    global _backend

    backend = _validate_backend(backend)
    if backend == "cupy":
        _import_cupy()
    _backend = backend


def get_array_module(backend: str | None = None) -> Any:
    """Return the active NumPy- or CuPy-compatible array module.

    Args:
        backend: Optional explicit backend. ``None`` and ``"auto"`` follow the
            global backend setting, which defaults to the legacy environment
            variable policy.

    Returns:
        ``numpy`` or ``cupy``.

    Raises:
        ImportError: If CuPy is explicitly requested but unavailable.
        ValueError: If the backend name is unsupported.
    """
    fallback_to_numpy = False
    if backend is None or backend == "auto":
        backend = _backend
        if backend == "auto":
            backend = _env_backend()
            fallback_to_numpy = True
    backend = _validate_backend(backend)

    if backend == "numpy":
        return np
    if backend == "cupy":
        try:
            return _import_cupy()
        except ImportError:
            if fallback_to_numpy:
                return np
            raise
    return np


def get_backend() -> str:
    """Return the currently active concrete backend name.

    Returns:
        Concrete backend name, either ``"numpy"`` or ``"cupy"``.
    """
    return "cupy" if get_array_module().__name__ == "cupy" else "numpy"


def asnumpy(array: Any) -> np.ndarray:
    """Return ``array`` as a NumPy array, copying from the GPU when needed.

    Args:
        array: NumPy-like or CuPy-like array.

    Returns:
        Host-side NumPy array.
    """
    module = get_array_module()
    if module is not np and hasattr(module, "asnumpy"):
        return module.asnumpy(array)
    return np.asarray(array)


class _ArrayModuleProxy:
    """Late-bound array module proxy used by voltage internals."""

    @property
    def __name__(self) -> str:
        """Return the active array module name.

        Returns:
            Active module name.
        """
        return get_array_module().__name__

    def __getattr__(self, name: str) -> Any:
        """Forward attribute lookups to the active array module.

        Args:
            name: Attribute name to resolve.

        Returns:
            Attribute from the active array module.
        """
        return getattr(get_array_module(), name)

    def __repr__(self) -> str:
        return repr(get_array_module())


xp = _ArrayModuleProxy()
