from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


def _copy_docstring(copy_func: Callable[..., Any]) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Copy a source docstring onto another callable.

    Args:
        copy_func: Callable whose docstring should be reused.

    Returns:
        Decorator that applies the docstring to the wrapped callable.
    """
    def wrapped(func: Callable[..., Any]) -> Callable[..., Any]:
        func.__doc__ = copy_func.__doc__
        return func

    return wrapped


def db(a: np.ndarray | float) -> np.ndarray | float:
    """Convert linear power values to decibels.

    Args:
        a: Linear power values.

    Returns:
        Values converted with ``10 * log10``.
    """
    return 10 * np.log10(a)


def array(fr: Any) -> np.ndarray:
    """Return the underlying NumPy array for a frame-like object.

    Args:
        fr: ``Frame`` instance or raw array-like input.

    Returns:
        Two-dimensional data array.
    """
    try:
        return fr.get_data()
    except AttributeError:
        return fr
