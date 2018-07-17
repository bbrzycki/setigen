import numpy as np

def constant_t_profile(level=1):
    def t_profile(t):
        return level
    return t_profile

def sine_t_profile(period, phase = 0, amplitude=1, level=1):
    t_profile = lambda t: amplitude * np.sin(2 * np.pi * (t + phase) / period) + level
    return t_profile

def periodic_gaussian_t_profile(period, phase, width, sigma, pnum = 1, amplitude = 1, level = 0):
    # width is width of individual pulses, sigma is variation in period
    def t_profile(t):
        center_k = np.round((t+phase)/period-1/4.)
        if pnum % 2 == 1:
            temp = (pnum - 1) / 2
            center_k = np.array([center_k + 1 * i for i in np.arange(-temp, temp + 1)])
        else:
            temp = pnum / 2
            center_k = np.array([center_k + 1 * i for i in np.arange(-temp + 1, temp + 1)])
        center = (4.*center_k+1.)/4.*period-phase

        # Calculate unique offsets per pulse and add to centers of Gaussians
        unique_centers = np.unique(center_k)
        mydict = dict(zip(unique_centers, np.random.normal(0,sigma,unique_centers.shape)))
        func = np.vectorize(lambda x: mydict[x])
        offsets = func(center_k)
        center += offsets

        intensity = 0
        for c in center:
            intensity -= stg.gaussian(t, c, width) * amplitude

        intensity += level
        return intensity
    return t_profile
