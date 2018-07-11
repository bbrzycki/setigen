import numpy as np
import sys

def db(x):
    """ Convert to dB """
    return 10*np.log10(x)

def gaussian_noise(data, mean, width):
    # Match data dimensions
    ts_len, fs_len = data.shape
    noise = np.random.normal(mean, width, [ts_len, fs_len])
    return noise

def normalize(data, exclude=0.0):
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
    if exclude == 1.0:
        sys.exit('Cannot exclude all data!')
    if exclude > 1.0 or exclude < 0.0:
        sys.exit('Invalid exclusion fraction! Must be between 0 and 1.')

    t_len, f_len = data.shape
    mean = np.empty(f_len)
    std = np.empty(f_len)
    for i in np.arange(f_len):
        noise_data = np.sort(data[:,i])[0:int(np.ceil(t_len*(1 - exclude)))]
        mean[i] = np.mean(noise_data)
        std[i] = np.std(noise_data)
    return np.nan_to_num((data - mean) / std)

def normalize_by_max(data):
    return data / np.max(data)

def inject_noise(data,
                 modulate_signal = False,
                 modulate_width = 0.1,
                 background_noise = True,
                 noise_width = 1):
    new_data = data
    if modulate_signal:
        new_data *= gaussian_noise(data, 1, modulate_width)
    if background_noise:
        new_data += gaussian_noise(data, 0, noise_width)
    return new_data
