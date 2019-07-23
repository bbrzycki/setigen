import numpy as np

from . import fil_utils
from . import split_utils
from . import stats


def sample_from_array(array):
    '''Take a random sample from a provided NumPy array'''
    return array[np.random.randint(0, len(array))]


def sample_gaussian_params(x_mean_array, x_std_array, x_min_array=None):
    x_mean = sample_from_array(x_mean_array)
    x_std = sample_from_array(x_std_array)

    # Somewhat arbitrary decision to ensure that the mean is at least the
    # standard deviation
    x_mean = np.maximum(x_mean, x_std)

    if x_min_array is not None:
        x_min = sample_from_array(x_min_array)
        return x_mean, x_std, x_min

    return x_mean, x_std


def get_parameter_distributions(fil_fn, f_window, f_shift=None, exclude=0):
    """
    Calculate parameter distributions for the mean, standard deviation,
    and minimum of split filterbank data from real observations.

    Parameters
    ----------
    fil_fn : str
        Filterbank filename with .fil extension
    f_window : int
        Number of frequency samples per new filterbank file
    f_shift : int, optional
        Number of samples to shift when splitting filterbank. If
        None, defaults to `f_shift=f_window` so that there is no
        overlap between new filterbank files

    Returns:
    --------
    (x_mean_array, x_std_array, x_min_array) : tuple of Numpy arrays
        Distributed of Gaussian parameters estimated from observations
    """
    split_generator = split_utils.split_fil_generator(fil_fn,
                                                      f_window,
                                                      f_shift=f_shift)

    x_mean_array = []
    x_std_array = []
    x_min_array = []
    for split_fil in split_generator:
        x_mean, x_std, x_min = stats.compute_frame_stats(fil_utils.get_data(split_fil),
                                                         exclude=exclude)
        x_mean_array.append(x_mean)
        x_std_array.append(x_std)
        x_min_array.append(x_min)
    x_mean_array = np.array(x_mean_array)
    x_std_array = np.array(x_std_array)
    x_min_array = np.array(x_min_array)

    return (x_mean_array, x_std_array, x_min_array)
