from __future__ import annotations

from pathlib import PurePath
import numpy as np
from blimpy import Waterfall

from ._typing import PathLike


def max_freq(waterfall: PathLike | Waterfall) -> float:
    """Return the highest channel-center frequency in a waterfall.

    Args:
        waterfall: Filterbank filename or waterfall object.

    Returns:
        Maximum frequency in the data.
    """
    return np.max(get_fs(waterfall))


def min_freq(waterfall: PathLike | Waterfall) -> float:
    """Return the lowest channel-center frequency in a waterfall.

    Args:
        waterfall: Filterbank filename or waterfall object.

    Returns:
        Minimum frequency in the data.
    """
    return np.min(get_fs(waterfall))


def get_data(waterfall: PathLike | Waterfall, db: bool = False) -> np.ndarray:
    """Return waterfall data as a two-dimensional array.

    Args:
        waterfall: Filterbank filename or waterfall object.
        db: Whether to convert intensities to dB.

    Returns:
        Time-frequency data array.

    Raises:
        ValueError: If the waterfall input type is unsupported.
    """
    if isinstance(waterfall, (str, PurePath)):
        waterfall = Waterfall(waterfall)
    elif not isinstance(waterfall, Waterfall):
        raise ValueError('Invalid data file!')

    if db:
        return 10 * np.log10(waterfall.data[:, 0, :])

    return waterfall.data[:, 0, :]


def get_fs(waterfall: PathLike | Waterfall) -> np.ndarray:
    """Return waterfall frequency values.

    Args:
        waterfall: Filterbank filename or waterfall object.

    Returns:
        Frequency axis in MHz.

    Raises:
        ValueError: If the waterfall input type is unsupported.
    """
    if isinstance(waterfall, (str, PurePath)):
        waterfall = Waterfall(waterfall, load_data=False)
    elif not isinstance(waterfall, Waterfall):
        raise ValueError('Invalid data file!')

    fch1 = waterfall.header['fch1']
    df = waterfall.header['foff']
    fchans = waterfall.header['nchans']

    return fch1 + np.arange(fchans) * df


def get_ts(waterfall: PathLike | Waterfall) -> np.ndarray:
    """Return waterfall time values.

    Args:
        waterfall: Filterbank filename or waterfall object.

    Returns:
        Time axis in seconds relative to the start.

    Raises:
        ValueError: If the waterfall input type is unsupported.
    """
    if isinstance(waterfall, (str, PurePath)):
        waterfall = Waterfall(waterfall, load_data=False)
    elif not isinstance(waterfall, Waterfall):
        raise ValueError('Invalid data file!')

    tsamp = waterfall.header['tsamp']
    tchans = waterfall.container.selection_shape[0]

    return np.arange(tchans) * tsamp
