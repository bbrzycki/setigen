import numpy as np
from astropy.stats import sigma_clip

from . import waterfall_utils
from . import split_utils


def sample_gaussian_params(x_mean_array, x_std_array, x_min_array=None):
    """
    Sample Gaussian parameters (mean, std, min) from provided arrays.

    Typical usage would be for select Gaussian noise properties for injection
    into data frames.

    Parameters
    ----------
    x_mean_array : ndarray
        Array of potential means
    x_std_array : ndarray
        Array of potential standard deviations
    x_min_array : ndarray, optional
        Array of potential minimum values

    Returns
    -------
    x_mean
        Selected mean of distribution
    x_std
        Selected standard deviation of distribution
    x_min
        If x_min_array provided, selected minimum of distribution
    """
    x_mean = np.random.choice(x_mean_array)
    x_std = np.random.choice(x_std_array)

    # Somewhat arbitrary decision to ensure that the mean is at least the
    # standard deviation
    x_mean = np.maximum(x_mean, x_std)

    if x_min_array is not None:
        x_min = np.random.choice(x_min_array)
        return x_mean, x_std, x_min

    return x_mean, x_std


def get_parameter_distributions(waterfall_fn, fchans, tchans=None, f_shift=None):
    """
    Calculate parameter distributions for the mean, standard deviation,
    and minimum of split filterbank data from real observations.

    Parameters
    ----------
    waterfall_fn : str
        Filterbank filename with .fil extension
    fchans : int
        Number of frequency samples per new filterbank file
    tchans : int, optional
        Number of time samples to select - will default from start of observation.
        If None, just uses the entire integration time
    f_shift : int, optional
        Number of samples to shift when splitting filterbank. If
        None, defaults to `f_shift=f_window` so that there is no
        overlap between new filterbank files

    Returns
    -------
    x_mean_array
        Distribution of means calculated from observations
    x_std_array
        Distribution of standard deviations calculated from observations
    x_min_array
        Distribution of minimums calculated from observations
    """
    split_generator = split_utils.split_waterfall_generator(waterfall_fn,
                                                            fchans,
                                                            tchans=tchans,
                                                            f_shift=f_shift)

    x_mean_array = []
    x_std_array = []
    x_min_array = []
    for waterfall in split_generator:
        clipped_data = sigma_clip(waterfall_utils.get_data(waterfall),
                                  sigma=3,
                                  maxiters=5,
                                  masked=False)
        x_mean_array.append(np.mean(clipped_data))
        x_std_array.append(np.std(clipped_data))
        x_min_array.append(np.min(clipped_data))

    x_mean_array = np.array(x_mean_array)
    x_std_array = np.array(x_std_array)
    x_min_array = np.array(x_min_array)

    return (x_mean_array, x_std_array, x_min_array)


def get_mean_distribution(waterfall_fn, fchans, tchans=None, f_shift=None):
    """
    Calculate parameter distributions for the mean of split filterbank frames 
    from real observations.

    Parameters
    ----------
    waterfall_fn : str
        Filterbank filename with .fil extension
    fchans : int
        Number of frequency samples per new filterbank file
    tchans : int, optional
        Number of time samples to select - will default from start of observation.
        If None, just uses the entire integration time
    f_shift : int, optional
        Number of samples to shift when splitting filterbank. If
        None, defaults to `f_shift=f_window` so that there is no
        overlap between new filterbank files

    Returns
    -------
    x_mean_array
        Distribution of means calculated from observations
    """
    split_generator = split_utils.split_waterfall_generator(waterfall_fn,
                                                            fchans,
                                                            tchans=tchans,
                                                            f_shift=f_shift)

    x_mean_array = []
    for waterfall in split_generator:
        clipped_data = sigma_clip(waterfall_utils.get_data(waterfall),
                                  sigma=3,
                                  maxiters=5,
                                  masked=False)
        x_mean_array.append(np.mean(clipped_data))

    x_mean_array = np.array(x_mean_array)

    return x_mean_array
