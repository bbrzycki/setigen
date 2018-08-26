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

def choppy_rfi_path(f_start, drift_rate, spread, spread_type='uniform'):
    def path(t):
        if spread_type == 'uniform':
            f_offset = np.random.uniform(-spread / 2., spread / 2.)
        elif spread_type == 'gaussian':
            f_offset = np.random.normal(0, spread)
        else:
            sys.exit('Wups')
        return f_start + drift_rate * t + f_offset
    return path
