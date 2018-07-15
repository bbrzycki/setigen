import numpy as np

def constant_t_profile(level=1):
    def t_profile(t):
        return level
    return t_profile

def sine_t_profile(period, amplitude=1, level=1):
    t_profile = lambda t: amplitude * np.sin(2*np.pi*t/period) + level
    return t_profile

def periodic_gaussian_t_profile(period, phase, sigma, pnum = 1, amplitude = 1, level = 0):
    def t_profile(t):
        center_k = np.round((t+phase)/period*2-1/2)
        if pnum % 2 == 1:
            temp = (pnum - 1) / 2
            center_k = np.array([center_k + i for i in np.arange(-temp, temp + 1)])
        else:
            temp = pnum / 2
            center_k = np.array([center_k + i for i in np.arange(-temp + 1, temp + 1)])
        center = (2*center_k+1.)/4*period-phase

        intensity = 0
        for c in center:
            intensity += stg.gaussian(t, c, sigma) * amplitude
        intensity += level
        return intensity
    return t_profile
