from __future__ import annotations

import numpy as np
from astropy.stats import median_absolute_deviation, sigma_clip
from . import utils
from .frame import Frame


def sigma_clip_norm(
    fr: Frame | np.ndarray,
    axis: str | int | None = None,
    background: Frame | np.ndarray | None = None,
) -> Frame | np.ndarray:
    """Normalize data using sigma-clipped background statistics.

    Args:
        fr: Input frame or array to normalize.
        axis: Optional axis along which to normalize.
        background: Optional background frame or array used for statistics.

    Returns:
        Normalized data with the same container type as the input.
    """
    data = utils.array(fr)
    
    if background is None:
        # If `data` is a Frame object, just use its data
        background = data
    else:
        background = utils.array(background)
        
    if axis in ['f', 1]:
        axis = 1
    elif axis in ['t', 0]:
        axis = 0

    clipped_data = sigma_clip(background, axis=axis, masked=True)
    data = data - np.mean(clipped_data, axis=axis, keepdims=True)
    data = data / np.std(clipped_data, axis=axis, keepdims=True)
    
    if isinstance(fr, Frame):
        n_frame = fr.copy()
        n_frame.data = data
        n_frame._update_noise_frame_stats()
        return n_frame
    else:
        return data
    

def sliding_norm(
    data: np.ndarray,
    cols: int = 0,
    exclude: float = 0.0,
    db: bool = False,
    use_median: bool = False,
) -> np.ndarray:
    """Normalize data per frequency channel using a sliding window.

    Args:
        data: Time-frequency data array.
        cols: Number of columns on either side of the current frequency bin.
        exclude: Fraction of brightest samples to exclude.
        db: Whether to convert data to dB before normalization.
        use_median: Whether to use median and MAD instead of mean and standard
            deviation.

    Returns:
        Normalized data array.
    """

    # Width of normalization window = 2 * cols + 1
    t_len, f_len = data.shape
    mean = np.empty(f_len)
    std = np.empty(f_len)
    if db:
        data = utils.db(data)
    for i in np.arange(f_len):
        if i < cols:
            start = 0
        else:
            start = i - cols
        if i > f_len - 1 - cols:
            end = f_len
        else:
            end = i + cols + 1
        temp = np.sort(data[:, start:end].flatten())
        noise = temp[0:int(np.ceil(t_len * (end - start) * (1 - exclude)))]
        if use_median:
            mean[i] = np.median(noise)
            std[i] = median_absolute_deviation(noise)
        else:
            mean[i] = np.mean(noise)
            std[i] = np.std(noise)
    return np.nan_to_num((data - mean) / std)


def blimpy_clip(data: np.ndarray, exclude: float = 0) -> np.ndarray:
    """Clip the brightest fraction of samples from a data array.

    Args:
        data: Time-frequency data array.
        exclude: Fraction of brightest samples to exclude.

    Returns:
        Clipped one-dimensional data array.
    """
    flat_data = data.flatten()
    return np.sort(flat_data)[::-1][int(exclude * len(flat_data)):]


def max_norm(data: np.ndarray) -> np.ndarray:
    """Normalize data by its maximum value.

    Args:
        data: Time-frequency data array.

    Returns:
        Maximum-normalized data.
    """
    return data / np.max(data)
