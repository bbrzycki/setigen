import numpy as np


def gaussian(x_mean, x_std, shape):
    return np.random.normal(x_mean, x_std, shape)


def truncated_gaussian(x_mean, x_std, x_min, shape):
    """
    Samples from a normal distribution, but enforces a minimum value.
    """
    return np.maximum(gaussian(x_mean, x_std, shape), x_min)
