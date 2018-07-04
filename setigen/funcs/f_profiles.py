import numpy as np

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def box_f_profile(width = 0.00001):
    def f_profile(f, f_center):
        return (np.abs(f-f_center) < width).astype(int)
    return f_profile

def gaussian_f_profile(width = 0.00001):
    def f_profile(f, f_center):
        return gaussian(f, f_center, width)
    return f_profile

def multiple_gaussian_f_profile(width = 0.00001):
    def f_profile(f, f_center):
        return gaussian(f, f_center - 0.0001, width)/4 \
            + gaussian(f, f_center, width) \
            + gaussian(f, f_center + 0.0001, width)/4
    return f_profile
