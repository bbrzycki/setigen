import sys
import numpy as np
from astropy import units as u

from setigen import unit_utils
from setigen.funcs import func_utils


def constant_t_profile(level=1):
    def t_profile(t):
        if type(t) in [np.ndarray, list]:
            shape = np.array(t).shape
        else:
            return level
        return np.full(shape, level)
    return t_profile


def sine_t_profile(period, phase=0, amplitude=1, level=1):
    period = unit_utils.get_value(period, u.s)

    def t_profile(t):
        return amplitude * np.sin(2 * np.pi * (t + phase) / period) + level
    return t_profile


def periodic_gaussian_t_profile(period,
                                phase,
                                pulse_offset_sigma,
                                pulse_width,
                                pulse_direction='rand',
                                pnum=1,
                                amplitude=1,
                                level=0):
    # pulse_direction can be 'up', 'down', 'rand'
    # pulse_offset_sigma is the variation in the pulse period, pulse_width is
    # the width of individual pulses; both are modeled as Guassians
    period = unit_utils.get_value(period, u.s)
    pulse_offset_sigma = unit_utils.get_value(pulse_offset_sigma, u.s)
    pulse_width = unit_utils.get_value(pulse_width, u.s)

    def t_profile(t):
        center_ks = np.round((t + phase) / period - 1 / 4.)
        if pnum % 2 == 1:
            temp = (pnum - 1) / 2
            center_ks = np.array([center_ks + 1 * i
                                  for i in np.arange(-temp, temp + 1)])
        else:
            temp = pnum / 2
            center_ks = np.array([center_ks + 1 * i
                                  for i in np.arange(-temp + 1, temp + 1)])
        centers = (4. * center_ks + 1.) / 4. * period - phase

        # Calculate unique offsets per pulse and add to centers of Gaussians
        unique_centers = np.unique(center_ks)

        offset_dict = dict(zip(unique_centers,
                               np.random.normal(0,
                                                pulse_offset_sigma,
                                                unique_centers.shape)))
        get_offsets = np.vectorize(lambda x: offset_dict[x])

        sign_list = []
        for c in unique_centers:
            x = np.random.uniform(0, 1)
            if (pulse_direction == 'up'
                    or pulse_direction == 'rand' and x < 0.5):
                sign_list.append(1)
            elif pulse_direction == 'down' or pulse_direction == 'rand':
                sign_list.append(-1)
            else:
                sys.exit('Invalid pulse direction!')
        sign_dict = dict(zip(unique_centers, sign_list))
        get_signs = np.vectorize(lambda x: sign_dict[x])

        centers += get_offsets(center_ks)
        center_signs = zip(centers, get_signs(center_ks))

        intensity = 0
        for c, sign in center_signs:
            intensity += sign * amplitude * func_utils.gaussian(t,
                                                                c,
                                                                pulse_width)

        intensity += level
        relu = np.vectorize(lambda x: max(0, x))
        return relu(intensity)
    return t_profile
