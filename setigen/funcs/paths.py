import numpy as np

def constant_path(f_start, drift_rate):
    def path(t):
        return f_start + drift_rate * t
    return path

def squared_path(f_start, drift_rate):
    def path(t):
        return f_start + drift_rate * t**2
    return path

def sine_path(f_start, drift_rate, period, amplitude):
    def path(t):
        return f_start + amplitude * np.sin(2*np.pi*t/period) + drift_rate * t
    return path
