import numpy as np


def sample_from_array(array):
    '''Take a random sample from a provided NumPy array'''
    return array[np.random.randint(0, len(array))]


def sample_gaussian_params(x_mean_array, x_std_array, x_min_array=None):
    x_mean = sample_from_array(x_mean_array)
    x_std = sample_from_array(x_std_array)
    
    # Somewhat arbitrary decision to ensure that the mean is at least the standard deviation
    x_mean = np.maximum(x_mean, x_std)
    
    if x_min_array is not None:
        x_min = sample_from_array(x_min_array)
        return x_mean, x_std, x_min
    
    return x_mean, x_std