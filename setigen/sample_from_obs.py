import numpy as np


def sample_from_array(array):
    '''Take a random sample from a provided NumPy array'''
    return array[np.random.randint(0, len(array))]


def sample_gaussian_params(x_mean_array, x_std_array, x_min_array=None):
    x_means = sample_from_array(x_mean_array)
    x_stds = sample_from_array(x_std_array)
    
    # Somewhat arbitrary decision to ensure that the mean is at least the standard deviation
    x_means = np.maximum(means, stds)
    
    if x_min_array:
        x_mins = sample_from_array(x_min_array)
        return x_means, x_stds, x_mins
    
    return x_means, x_stds