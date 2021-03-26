import os

GPU_FLAG = os.getenv('SETIGEN_ENABLE_GPU', '0')
if GPU_FLAG == '1':
    try:
        import cupy as xp
    except ImportError:
        import numpy as xp
else:
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
        
    def get_total_noise_std(self):
        return xp.sqrt(self.noise_std**2 + self.bg_noise_std**2)
    
    def add_noise(self, v_mean, v_std):
        noise_func = lambda ts: self.rng.normal(loc=v_mean, 
                                                scale=v_std,
                                                size=len(ts))
        # Variances add, not standard deviations
        self.noise_std = xp.sqrt(self.noise_std**2 + v_std**2)
        self.noise_sources.append(noise_func)
        
    def add_constant_signal(self,
                            f_start, 
                            drift_rate,
                            level,
                            phase=0):
        """
        phase is in radians
        """
        f_start = unit_utils.get_value(f_start, u.Hz)
        drift_rate = unit_utils.get_value(drift_rate, u.Hz / u.s)
        
        def signal_func(ts):
            # Calculate adjusted center frequencies, according to chirp
            chirp_phase = 2 * xp.pi * ((f_start - self.fch1) * ts + 0.5 * drift_rate * ts**2)
            if not self.ascending:
                chirp_phase = -chirp_phase
            return level * xp.cos(chirp_phase + phase)
        
        self.signal_sources.append(signal_func)
        
    def add_signal(self, signal_func):
        self.signal_sources.append(signal_func)
    
    def get_samples(self, num_samples):
        self._update_t(num_samples)
        
        for noise_func in self.noise_sources:
            self.v += noise_func(self.ts)
            
        for signal_func in self.signal_sources:
            self.v += signal_func(self.ts)  

        self.start_obs = False
        
        return self.v
        
        
class BackgroundDataStream(DataStream):
    """
    Extends DataStream for background data in Antenna arrays.
    """
    
    def __init__(self,
                 sample_rate=3*u.GHz,
                 fch1=0*u.GHz,
                 ascending=True,
                 t_start=0,
                 seed=None,
                 antenna_streams=[]):
        super().__init__(sample_rate=sample_rate,
                         fch1=fch1,
                         ascending=ascending,
                         t_start=t_start,
                         seed=seed)
        self.antenna_streams = antenna_streams
        
    def add_noise(self, v_mean, v_std):
        noise_func = lambda ts: self.rng.normal(loc=v_mean, 
                                                scale=v_std,
                                                size=len(ts))
        # Variances add, not standard deviations
        self.noise_std = xp.sqrt(self.noise_std**2 + v_std**2)
        for stream in self.antenna_streams:
            stream.bg_noise_std = xp.sqrt(stream.bg_noise_std**2 + v_std**2)
        self.noise_sources.append(noise_func)
        