import numpy as np
import sys

def normalize(data, exclude=0.0):
    """
    Normalize data per frequency channel so that the noise level in data is
    controlled. Excludes a fraction of brightest pixels to better isolate noise.

    Args:
        data, NumPy array with time-frequency data
        exclude, fraction of brightest samples in each frequency bin to exclude
            in calculating mean and standard deviation

    Return:
        normalized_data, NumPy array

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
