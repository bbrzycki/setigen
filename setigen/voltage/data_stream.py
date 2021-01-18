import sys
import os.path
import numpy as np
import matplotlib.pyplot as plt

from astropy import units as u
from astropy.stats import sigma_clip

from setigen import unit_utils


class DataStream(object):
    """
    Facilitates the creation of synthetic raw voltage data.
    """
    
    def __init__(self,
                 num_samples,
                 sample_rate=3*u.GHz):
        self.num_samples = num_samples
        self.sample_rate = unit_utils.get_value(sample_rate, u.Hz)
        self.dt = 1 / self.sample_rate
        
        self.x = np.linspace(0., 
                             self.num_samples * self.dt,
                             self.num_samples,
                             endpoint=False)
        self.y = np.zeros(self.num_samples)
    
    def add_noise(self,
                  x_mean,
                  x_std):
        noise = np.random.normal(loc=x_mean, 
                                  scale=x_std,
                                  size=self.num_samples)
        self.y += noise
        return noise
        
    def add_signal(self,
                   f_start, 
                   drift_rate,
                   level):
        f_start = unit_utils.get_value(f_start, u.Hz)
        drift_rate = unit_utils.get_value(drift_rate, u.Hz / u.s)
        
        center_freqs = f_start + drift_rate * self.x
        signal = level * np.cos(2 * np.pi * self.x * center_freqs)
        
        self.y += signal
        return signal