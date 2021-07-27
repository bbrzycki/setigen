import numpy as np
from astropy.stats import median_absolute_deviation, sigma_clip
from . import frame_utils
from .frame import Frame


def sigma_clip_norm(fr, axis=None, as_data=None):
    """
    Normalize data by subtracting out noise background, determined by
    sigma clipping.
    
    Parameters
    ----------
    fr : Frame or ndarray
        Input data to be normalized
    axis : int
        Axis along which data should be normalized. If None, will
        compute statistics over the entire data frame. 
    as_data : Frame or ndarray
        Data to be used for noise calculations
        
    Returns
    -------
    n_data
        Returns normalized data in the same type as f
    """
    data = frame_utils.array(fr)
    
    if as_data is None:
        # If `data` is a Frame object, just use its data
        as_data = data
    else:
        as_data = frame_utils.array(as_data)
        
    if axis in ['f', 1]:
        axis = 1
    else:
        axis = 0

    clipped_data = sigma_clip(as_data, axis=axis, masked=True)
    data = data - np.mean(clipped_data, axis=axis, keepdims=True)
    data = data / np.std(clipped_data, axis=axis, keepdims=True)
    
    if isinstance(fr, Frame):
        n_frame = fr.copy()
        n_frame.data = data
        return n_frame
    else:
        return data
    

def sliding_norm(data, cols=0, exclude=0.0, to_db=False, use_median=False):
    """
    Normalize data per frequency channel so that the noise level in data is
    controlled; using mean or median filter.

    Uses a sliding window to calculate mean and standard deviation
    to preserve non-drifted signals. Excludes a fraction of brightest pixels to
    better isolate noise.

    Parameters
    ----------
    data : ndarray
        Time-frequency data
    cols : int
        Number of columns on either side of the current frequency bin. The
        width of the sliding window is thus 2 * cols + 1
    exclude : float, optional
        Fraction of brightest samples in each frequency bin to exclude in
        calculating mean and standard deviation
    to_db : bool, optional
        Convert values to decibel equivalents *before* normalization
    use_median : bool, optional
        Use median and median absolute deviation instead of mean and standard
        deviation

    Returns
    -------
    normalized_data : ndarray
        Normalized data

    """

    # Width of normalization window = 2 * cols + 1
    t_len, f_len = data.shape
    mean = np.empty(f_len)
    std = np.empty(f_len)
    if to_db:
        data = db(data)
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


def max_norm(data):
    """
    Simple normalization by dividing out by the brightest pixel.
    """
    return data / np.max(data)
