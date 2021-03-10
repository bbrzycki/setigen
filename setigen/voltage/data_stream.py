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
                 fch1=0*u.GHz,
                 ascending=True,
                 t_start=0,
                 seed=None):
        self.rng = xp.random.RandomState(seed)
        
        self.sample_rate = unit_utils.get_value(sample_rate, u.Hz)
        self.dt = 1 / self.sample_rate
        
        # For adjusting signal frequencies
        self.fch1 = unit_utils.get_value(fch1, u.Hz)
        self.ascending = ascending
        
        # For estimating SNR for signals
        self.noise_std = 0
        self.bg_noise_std = 0
#         self.total_obs_num_samples = 0
        
        # Tracks start time of next sequence of data
        self.t_start = t_start
        self.start_obs = True
        
        # Hold functions that generate voltage values
        self.noise_sources = []
        self.signal_sources = []
        
    def _update_t(self, num_samples):
        self.ts = self.t_start + xp.linspace(0., 
                                             num_samples * self.dt,
                                             num_samples,
                                             endpoint=False)
        self.t_start += num_samples * self.dt
        self.v = xp.zeros(num_samples)
        
    def set_time(self, t):
        self.start_obs = True
        self.t_start = t
        
    def add_time(self, t):
        self.set_time(self.t_start + t)
    
    def add_noise(self, v_mean, v_std):
        noise_func = lambda ts: self.rng.normal(loc=v_mean, 
                                                scale=v_std,
                                                size=len(ts))
        self.noise_std += v_std
        self.noise_sources.append(noise_func)
        
    def add_constant_signal(self,
                            f_start, 
                            drift_rate,
                            level=None,
                            snr=None,
                            phase=0,
                            mode='level'):
        """
        mode can be 'level' or 'snr'
        """
        f_start = unit_utils.get_value(f_start, u.Hz)
        drift_rate = unit_utils.get_value(drift_rate, u.Hz / u.s)
        
        def signal_func(ts, total_obs_num_samples=None):
            # Calculate adjusted center frequencies
            center_freqs = f_start - self.fch1 + drift_rate * ts
            if not self.ascending:
                center_freqs = -center_freqs
                
            if mode == 'level':
                if level is None:
                    raise ValueError("Value not given for 'level'.")
                amplitude = level
            elif mode == 'snr':
                if snr is None:
                    raise ValueError("Value not given for 'snr'.")
                if self.noise_std + self.bg_noise_std == 0:
                    raise ValueError("No noise added to data stream.")
                assert total_obs_num_samples > 0
                amplitude = (snr * (self.noise_std + self.bg_noise_std) / xp.sqrt(total_obs_num_samples))**0.5
#                 amplitude = snr * (self.noise_std + self.bg_noise_std) / xp.sqrt(total_obs_num_samples)
                print(amplitude)
            else:
                raise ValueError("Invalid option given for 'mode'.")    
                
            return amplitude * xp.cos(2 * xp.pi * ts * center_freqs + phase)
        
        self.signal_sources.append(signal_func)
        
    def add_signal(self, signal_func):
        def new_signal_func(ts, total_obs_num_samples=None):
            return signal_func(ts)
        self.signal_sources.append(new_signal_func)
    
    def get_samples(self,
                    num_samples,
                    total_obs_num_samples=None):
        if total_obs_num_samples is not None:
            self.total_obs_num_samples = total_obs_num_samples
            
        self._update_t(num_samples)
        
        for noise_func in self.noise_sources:
            self.v += noise_func(self.ts)
            
        for signal_func in self.signal_sources:
            self.v += signal_func(self.ts, total_obs_num_samples=total_obs_num_samples)  

        self.start_obs = False
        
        return self.v
        