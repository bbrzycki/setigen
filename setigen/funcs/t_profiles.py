import numpy as np

def constant_t_profile(level=1):
    def t_profile(t):
        return level
    return t_profile

def sine_t_profile(period, amplitude=1, level=1):
    t_profile = lambda t: amplitude * np.sin(2*np.pi*t/period) + level
    return t_profile
