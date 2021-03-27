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
        self.ts = None
        self.v = None
        
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
        
    def update_noise(self, stats_calc_num_samples=10000):
        """
        Replace self.noise_std by calculating out a few samples and estimating the 
        standard deviation of the voltages.
        """
        start_obs = self.start_obs
        t_start = self.t_start
        
        v = self.get_samples(num_samples=stats_calc_num_samples)
        _, self.noise_std = estimate_stats(v, stats_calc_num_samples=stats_calc_num_samples)
        
        self.start_obs = start_obs
        self.t_start = t_start
        
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
            # Ensure that the array is of the correct type
            self.v += xp.array(signal_func(self.ts))
            
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
        
    def _set_all_bg_noise(self):
        for stream in self.antenna_streams:
            stream.bg_noise_std = self.noise_std
        
    def update_noise(self, stats_calc_num_samples=10000):
        """
        Replace self.noise_std by calculating out a few samples and estimating the 
        standard deviation of the voltages.
        """
        DataStream.update_noise(self, stats_calc_num_samples=stats_calc_num_samples)
        self._set_all_bg_noise()
        
    def add_noise(self, v_mean, v_std):
        DataStream.add_noise(self, v_mean, v_std)
        self._set_all_bg_noise()
        
        
def estimate_stats(voltages, stats_calc_num_samples=10000):
    """
    Estimate mean and standard deviation, truncating to at most `stats_calc_num_samples` samples 
    to reduce computation.
    """
    calc_len = xp.amin(xp.array([stats_calc_num_samples, len(voltages)]))
    data_sigma = xp.std(voltages[:calc_len])
    data_mean = xp.mean(voltages[:calc_len])
    
    return data_mean, data_sigma