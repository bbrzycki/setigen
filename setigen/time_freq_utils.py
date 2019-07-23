import numpy as np
from astropy.stats import median_absolute_deviation


def db(x):
    """ Convert to dB """
    return 10 * np.log10(x)


def choose_from_dist(dist, shape):
    """Load random values from a loaded NumPy array in the specified shape"""
    return dist[np.random.randint(0, len(dist), shape)]


def make_normal(means_dist, stds_dist, mins_dist, shape):
    """
    Grab means, standard deviations, and minimums from the loaded distributions
    (each NumPy arrays) in the shape provided.
    """
    means = choose_from_dist(means_dist, shape)
    stds = choose_from_dist(stds_dist, shape)
    mins = choose_from_dist(mins_dist, shape)
    # means = np.maximum(means, stds)
    return means, stds, mins


def gaussian_frame_from_dist(means_dist, stds_dist, mins_dist, shape):
    """
    Make a Gaussian noise frame from given distributions for
    the mean, standard deviation, and minimums for data in the shape provided.
    """
    mean, std, minimum = make_normal(means_dist, stds_dist, mins_dist, 1)
    return np.maximum(np.random.normal(mean, std, shape),
                      minimum), mean, std, minimum


def normalize(data, cols=0, exclude=0.0, to_db=False, use_median=False):
    """Normalize data per frequency channel so that the noise level in data is
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
    """Simple normalization by dividing out by the brightest pixel"""
    return data / np.max(data)


def inject_noise(data,
                 modulate_signal=False,
                 modulate_width=0.1,
                 background_noise=True,
                 noise_sigma=1):
    """Normalize data per frequency channel so that the noise level in data is
    controlled.

    Uses a sliding window to calculate mean and standard deviation
    to preserve non-drifted signals. Excludes a fraction of brightest pixels to
    better isolate noise.

    Parameters
    ----------
    data : ndarray
        Time-frequency data
    modulate_signal : bool, optional
        Modulate signal itself with Gaussian noise (multiplicative)
    modulate_width : float, optional
        Standard deviation of signal modulation
    background_noise : bool, optional
        Add gaussian noise to entire image (additive)
    noise_sigma : float, optional
        Standard deviation of background Gaussian noise

    Returns
    -------
    noisy_data : ndarray
        Data with injected noise

    """
    ts_len, fs_len = data.shape
    new_data = data
    if modulate_signal:
        new_data = new_data * np.random.normal(1,
                                               modulate_width,
                                               [ts_len, fs_len])
    if background_noise:
        new_data = new_data + np.random.normal(0,
                                               noise_sigma,
                                               [ts_len, fs_len])
    return new_data
