import numpy as np
from astropy.stats import median_absolute_deviation
import sys

def db(x):
    """ Convert to dB """
    return 10*np.log10(x)

def gaussian_noise(data, mean, sigma):
    # Match data dimensions
    ts_len, fs_len = data.shape
    noise = np.random.normal(mean, sigma, [ts_len, fs_len])
    return noise

def normalize(data, cols=0, exclude=0.0, to_db=False, use_median=False):
    """Normalize data per frequency channel so that the noise level in data is
    controlled. Excludes a fraction of brightest pixels to better isolate noise.

    Parameters
    ----------
    data : ndarray
        Time-frequency data
    exclude : float, optional
        Fraction of brightest samples in each frequency bin to exclude in
        calculating mean and standard deviation

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
        noise_data = np.sort(data[:,start:end].flatten())[0:int(np.ceil(t_len * (end - start)(1 - exclude)))]
        if use_median:
            mean[i] = np.median(noise_data)
            std[i] = median_absolute_deviation(noise_data)
        else:
            mean[i] = np.mean(noise_data)
            std[i] = np.std(noise_data)
    return np.nan_to_num((data - mean) / std)

def normalize_by_max(data):
    return data / np.max(data)

def inject_noise(data,
                 modulate_signal = False,
                 modulate_width = 0.1,
                 background_noise = True,
                 noise_sigma = 1):
    new_data = data
    if modulate_signal:
        new_data = new_data * gaussian_noise(data, 1, modulate_width)
    if background_noise:
        new_data = new_data + gaussian_noise(data, 0, noise_sigma)
    return new_data
