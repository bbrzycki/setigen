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
                 sample_rate=3*u.GHz):
        self.sample_rate = unit_utils.get_value(sample_rate, u.Hz)
        self.dt = 1 / self.sample_rate
        
        self.t_start = 0
        self.next_t_start = 0
        
        # Hold functions that generate voltage values
        self.noise_sources = []
        self.signal_sources = []
        
    def _update_t(self, num_samples):
        self.t = self.next_t_start + np.linspace(0., 
                                            num_samples * self.dt,
                                            num_samples,
                                            endpoint=False)
        
        self.t_start = self.t[0]
        self.next_t_start = self.t[-1] + self.dt
        
        self.v = np.zeros(num_samples)
    
    def add_noise(self,
                  v_mean,
                  v_std):
        noise_func = lambda t: np.random.normal(loc=v_mean, 
                                                scale=v_std,
                                                size=len(t))
        self.noise_sources.append(noise_func)
        
    def add_signal(self,
                   f_start, 
                   drift_rate,
                   level):
        f_start = unit_utils.get_value(f_start, u.Hz)
        drift_rate = unit_utils.get_value(drift_rate, u.Hz / u.s)
        
        def signal_func(t):
            center_freqs = f_start + drift_rate * self.t
            return level * np.cos(2 * np.pi * t * center_freqs)
        
        self.signal_sources.append(signal_func)
    
    def get_samples(self, 
                    num_samples):
        self._update_t(num_samples)
        for noise_func in self.noise_sources:
            self.v += noise_func(self.t)
        for signal_func in self.signal_sources:
            self.v += signal_func(self.t)   
        return self.v
        