import numpy as np
from astropy.stats import median_absolute_deviation


def db(x):
    """
    Converts to dB.
    """
    return 10 * np.log10(x)


def normalize(data, cols=0, exclude=0.0, to_db=False, use_median=False):
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


def normalize_by_max(data):
    """
    Simple normalization by dividing out by the brightest pixel.
    """
    return data / np.max(data)
