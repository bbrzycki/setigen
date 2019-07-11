import numpy as np


def get_mean(data, exclude=0):
    flat_data = data.flatten()
    excluded_flat_data = np.sort(flat_data)[::-1][int(exclude * len(flat_data)):]
    return np.mean(excluded_flat_data)


def get_std(data, exclude=0):
    flat_data = data.flatten()
    excluded_flat_data = np.sort(flat_data)[::-1][int(exclude * len(flat_data)):]
    return np.std(excluded_flat_data)


def get_min(data, exclude=0):
    flat_data = data.flatten()
    excluded_flat_data = np.sort(flat_data)[::-1][int(exclude * len(flat_data)):]
    return np.min(excluded_flat_data)


def compute_frame_stats(data, exclude=0):
    flat_data = data.flatten()
    excluded_flat_data = np.sort(flat_data)[::-1][int(exclude * len(flat_data)):]
    return (np.mean(excluded_flat_data), 
            np.std(excluded_flat_data), 
            np.min(excluded_flat_data))