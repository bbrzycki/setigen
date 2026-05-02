from __future__ import annotations

import os
import errno
from pathlib import Path
import numpy as np
from blimpy import Waterfall
from typing import Iterator

from ._typing import PathLike
from ._frame.io import _close_waterfall_handles


def split_waterfall_generator(
    waterfall_fn: PathLike,
    fchans: int,
    tchans: int | None = None,
    f_shift: int | None = None,
) -> Iterator[Waterfall]:
    """Yield smaller waterfall views split from a larger filterbank file.

    Args:
        waterfall_fn: Input filterbank filename.
        fchans: Number of frequency samples per split.
        tchans: Optional number of time samples to include.
        f_shift: Optional shift in frequency bins between splits.

    Yields:
        Waterfall views covering smaller sections of the input.

    Raises:
        ValueError: If `tchans` exceeds the available number of time samples.
    """

    info_wf = Waterfall(waterfall_fn, load_data=False)
    try:
        fch1 = info_wf.header['fch1']
        nchans = info_wf.header['nchans']
        df = info_wf.header['foff']
        tchans_tot = info_wf.container.selection_shape[0]
    finally:
        _close_waterfall_handles(info_wf)

    if f_shift is None:
        f_shift = fchans

    if tchans is None:
        tchans = tchans_tot
    elif tchans > tchans_tot:
        raise ValueError('tchans value must be less than the total number of \
                          time samples in the observation')

    # Note that df is negative!
    f_start, f_stop = fch1, fch1 + fchans * df

    # Iterates down frequencies, starting from highest
    while np.abs(f_stop - fch1) <= np.abs(nchans * df):
        fmin, fmax = np.sort([f_start, f_stop])
        waterfall = Waterfall(waterfall_fn,
                              f_start=fmin,
                              f_stop=fmax,
                              t_start=0,
                              t_stop=tchans)

        try:
            yield waterfall
        finally:
            _close_waterfall_handles(waterfall)

        f_start += f_shift * df
        f_stop += f_shift * df


def split_fil(
    waterfall_fn: PathLike,
    output_dir: PathLike,
    fchans: int,
    tchans: int | None = None,
    f_shift: int | None = None,
) -> list[Path]:
    """Split a filterbank file into smaller `.fil` files.

    Args:
        waterfall_fn: Input filterbank filename.
        output_dir: Directory for the new filterbank files.
        fchans: Number of frequency samples per split file.
        tchans: Optional number of time samples to include.
        f_shift: Optional shift in frequency bins between splits.

    Returns:
        Paths to the new filterbank files.
    """
    output_dir = Path(output_dir)

    try:
        os.makedirs(output_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    split_generator = split_waterfall_generator(waterfall_fn,
                                                fchans,
                                                tchans=tchans,
                                                f_shift=f_shift)

    # Iterates down frequencies, starting from highest
    split_fns = []
    for i, waterfall in enumerate(split_generator):
        output_fn = output_dir / f"{fchans}_{i:04d}.fil"
        waterfall.write_to_fil(output_fn)
        split_fns.append(output_fn)
        print(f"Saved {output_fn}")
    return split_fns


def split_array(data: np.ndarray,
                f_sample_num: int | None = None,
                t_sample_num: int | None = None,
                f_shift: int | None = None,
                t_shift: int | None = None,
                f_trim: bool = False,
                t_trim: bool = False) -> np.ndarray:
    """Split an array into smaller frequency-time windows.

    Args:
        data: Two-dimensional time-frequency data array.
        f_sample_num: Number of frequency samples per split.
        t_sample_num: Number of time samples per split.
        f_shift: Shift in frequency bins between splits.
        t_shift: Shift in time bins between splits.
        f_trim: Whether to drop splits with incomplete frequency width.
        t_trim: Whether to drop splits with incomplete time height.

    Returns:
        Array of split time-frequency windows.

    Raises:
        ValueError: If the input is not an ndarray or an invalid shift is
            supplied.
    """

    split_data = []

    if not isinstance(data, np.ndarray):
        raise ValueError("Input data must be a numpy array")

    height, width = data.shape

    if f_sample_num is None:
        f_sample_num = width
    if t_sample_num is None:
        t_sample_num = height

    if f_shift is None:
        f_shift = f_sample_num
    elif f_shift <= 0:
        raise ValueError(f"Invalid x-direction shift: {f_shift}")

    if t_shift is None:
        t_shift = t_sample_num
    elif t_shift <= 0:
        raise ValueError(f"Invalid y-direction shift: {t_shift}")

    # Save first frame, regardless of overstepping bounds
    y_start = 0
    y_stop = min(t_sample_num, height)
    x_start = 0
    x_stop = min(f_sample_num, width)
    split_data.append(data[y_start:y_stop, x_start:x_stop])
    y_in_bound = (y_stop < height)
    x_in_bound = (x_stop < width)

    # As long as either bound is valid, continue adding frames
    while y_in_bound or x_in_bound:

        # Shift frames in the x direction
        while x_in_bound:
            x_start = x_start + f_shift
            x_stop = min(x_stop + f_shift, width)
            split_data.append(data[y_start:y_stop, x_start:x_stop])
            x_in_bound = (x_stop < width)

        # Break when both y and x are out of bounds
        if not y_in_bound:
            break

        # Shift frames in the y direction and reset x indices
        y_start = y_start + t_shift
        y_stop = min(y_stop + t_shift, height)
        x_start = 0
        x_stop = min(f_sample_num, width)
        split_data.append(data[y_start:y_stop, x_start:x_stop])
        y_in_bound = (y_stop < height)
        x_in_bound = (x_stop < width)

    # Filter out frames that aren't the same specied size
    if t_trim:
        split_data = list(filter(lambda A: A.shape[0] == t_sample_num,
                                 split_data))
    if f_trim:
        split_data = list(filter(lambda A: A.shape[1] == f_sample_num,
                                 split_data))
    return np.array(split_data)
