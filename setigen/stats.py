import numpy as np


def exclude_and_flatten(data, exclude=0):
    flat_data = data.flatten()
    return np.sort(flat_data)[::-1][int(exclude * len(flat_data)):]


def get_mean(data, exclude=0):
    return np.mean(exclude_and_flatten(data, exclude=exclude))


def get_std(data, exclude=0):
    return np.std(exclude_and_flatten(data, exclude=exclude))


def get_min(data, exclude=0):
    return np.min(exclude_and_flatten(data, exclude=exclude))


def compute_frame_stats(data, exclude=0):
    excluded_flat_data = exclude_and_flatten(data, exclude=exclude)
    return (np.mean(excluded_flat_data),
            np.std(excluded_flat_data),
            np.min(excluded_flat_data))
