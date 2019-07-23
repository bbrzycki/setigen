import numpy as np


def gaussian(x_mean, x_std, shape):
    return np.random.normal(x_mean, x_std, shape)


def truncated_gaussian(x_mean, x_std, x_min, shape):
    return np.maximum(gaussian(x_mean, x_std, shape), x_min)
