try:
    import cupy as xp
except ImportError:
    import numpy as xp

from astropy import units as u
from astropy.stats import sigma_clip

import time

from setigen import unit_utils


class DataStream(object):
    """
    Facilitates the creation of synthetic raw voltage data.
    """
    
    def __init__(self,
                 sample_rate=3*u.GHz,
                 seed=None):
        self.rng = xp.random.RandomState(seed)
        
        self.sample_rate = unit_utils.get_value(sample_rate, u.Hz)
        self.dt = 1 / self.sample_rate
        
        self.reset()
        
        # Hold functions that generate voltage values
        self.noise_sources = []
        self.signal_sources = []
        
    def _update_t(self, num_samples):
        self.t = self.next_t_start + xp.linspace(0., 
                                                 num_samples * self.dt,
                                                 num_samples,
                                                 endpoint=False)
        
        self.t_start = self.t[0]
        self.next_t_start = self.t[-1] + self.dt
        
        self.v = xp.zeros(num_samples)
        
    def reset(self):
        self.next_t_start = self.t_start = 0
        self.t = None
        self.v = None
        self.start_obs = True
        
    def set_time(self, t):
        self.next_t_start = self.t_start = t
        
    def add_time(self, t):
        self.next_t_start += t
        self.t_start += t
    
    def add_noise(self,
                  v_mean,
                  v_std):
        noise_func = lambda t: self.rng.normal(loc=v_mean, 
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
            return level * xp.cos(2 * xp.pi * t * center_freqs)
        
        self.signal_sources.append(signal_func)
    
    def get_samples(self,
                    num_samples):
        self._update_t(num_samples)
        
#         start = time.time()
        
        for noise_func in self.noise_sources:
            self.v += noise_func(self.t)
            
#         print('noise',time.time() - start)
#         start = time.time()
            
        for signal_func in self.signal_sources:
            self.v += signal_func(self.t)   
        
#         print('signal',time.time() - start)
#         start = time.time()

        self.start_obs = False
        
        return self.v
        