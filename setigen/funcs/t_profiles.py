import numpy as np
import func_utils

def constant_t_profile(level=1):
    def t_profile(t):
        return level
    return t_profile

def sine_t_profile(period, phase = 0, amplitude=1, level=1):
    t_profile = lambda t: amplitude * np.sin(2 * np.pi * (t + phase) / period) + level
    return t_profile

def periodic_gaussian_t_profile(period, phase, sigma, pulse_dir, width, pnum = 1, amplitude = 1, level = 0):
    # pulse_dir can be 'up', 'down', 'rand'
    # width is width of individual pulses, sigma is variation in period
    def t_profile(t):
        center_ks = np.round((t + phase) / period - 1 / 4.)
        if pnum % 2 == 1:
            temp = (pnum - 1) / 2
            center_ks = np.array([center_ks + 1 * i for i in np.arange(-temp, temp + 1)])
        else:
            temp = pnum / 2
            center_ks = np.array([center_ks + 1 * i for i in np.arange(-temp + 1, temp + 1)])
        centers = (4. * center_ks + 1.) / 4. * period - phase

        # Calculate unique offsets per pulse and add to centers of Gaussians
        unique_centers = np.unique(center_ks)

        offset_dict = dict(zip(unique_centers, np.random.normal(0, sigma, unique_centers.shape)))
        get_offsets = np.vectorize(lambda x: offset_dict[x])

        sign_list = []
        for c in unique_centers:
            x = np.random.uniform(0, 1)
            if pulse_dir == 'up' or pulse_dir == 'rand' and x < 0.5:
                sign_list.append(1)
            elif pulse_dir == 'down' or pulse_dir == 'rand':
                sign_list.append(-1)
            else:
                sys.exit('Invalid pulse direction!')
        sign_dict = dict(zip(unique_centers, sign_list))
        get_signs = np.vectorize(lambda x: sign_dict[x])

        centers += get_offsets(center_ks)
        center_signs = zip(centers, get_signs(center_ks))

        intensity = 0
        for c, sign in center_signs:
            intensity += sign * amplitude * func_utils.gaussian(t, c, width)

        intensity += level
        relu = np.vectorize(lambda x: max(0,x))
        return relu(intensity)
    return t_profile
