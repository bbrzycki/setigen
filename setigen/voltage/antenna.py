try:
    import cupy as xp
except ImportError:
    import numpy as xp

from astropy import units as u

from setigen import unit_utils
from . import data_stream


class Antenna(object):
    def __init__(self,
                 sample_rate=3*u.GHz,
                 fch1=0*u.GHz,
                 ascending=True,
                 num_pols=2,
                 t_start=0,
                 seed=None,
                 delay=0):
        self.rng = xp.random.RandomState(seed)
        self.delay = delay
        
        self.sample_rate = unit_utils.get_value(sample_rate, u.Hz)
        self.dt = 1 / self.sample_rate
        
        self.fch1 = unit_utils.get_value(fch1, u.Hz)
        self.ascending = ascending
        
        assert num_pols in [1, 2]
        self.num_pols = num_pols
        
        self.t_start = t_start
        self.start_obs = True
        
        self.x = data_stream.DataStream(sample_rate=self.sample_rate,
                                        fch1=self.fch1,
                                        ascending=self.ascending,
                                        t_start=self.t_start,
                                        seed=int(self.rng.randint(2**32)))
        self.streams = [self.x]
        
        if self.num_pols == 2:
            self.y = data_stream.DataStream(sample_rate=self.sample_rate,
                                            fch1=self.fch1,
                                            ascending=self.ascending,
                                            t_start=self.t_start,
                                            seed=int(self.rng.randint(2**32)))
            self.streams.append(self.y)
        
        self.bg_cache = [None, None]
        
    def set_time(self, t):
        self.start_obs = True
        self.t_start = t
        self.x.set_time(t)
        if self.num_pols == 2:
            self.y.set_time(t)
        
    def add_time(self, t):
        self.set_time(self.t_start + t)
        
    def reset_start(self):
        self.add_time(0)
        
    def get_samples(self, num_samples, total_obs_num_samples=None):
        if self.num_pols == 2:
            samples = [[self.x.get_samples(num_samples, 
                                           total_obs_num_samples=total_obs_num_samples), 
                        self.y.get_samples(num_samples, 
                                           total_obs_num_samples=total_obs_num_samples)]]
        else:
            samples = [[self.x.get_samples(num_samples, 
                                           total_obs_num_samples=total_obs_num_samples)]]
            
        self.t_start += num_samples * self.dt
        self.start_obs = False
        
        return samples

        
class MultiAntennaArray(object):
    def __init__(self,
                 num_antennas,
                 sample_rate=3*u.GHz,
                 fch1=0*u.GHz,
                 ascending=True,
                 num_pols=2,
                 delays=None,
                 t_start=0,
                 seed=None):
        self.rng = xp.random.RandomState(seed)
        if delays is None:
            self.delays = xp.zeros(num_antennas)
        else:
            assert len(delays) == num_antennas
            self.delays = xp.array(delays).astype(int)
        self.max_delay = int(xp.max(delays))
        
        self.num_antennas = num_antennas
        self.sample_rate = unit_utils.get_value(sample_rate, u.Hz)
        self.dt = 1 / self.sample_rate
        
        self.fch1 = unit_utils.get_value(fch1, u.Hz)
        self.ascending = ascending
        
        assert num_pols in [1, 2]
        self.num_pols = num_pols
        
        self.t_start = t_start
        self.start_obs = True
        
        self.bg_x = data_stream.DataStream(sample_rate=self.sample_rate,
                                           fch1=self.fch1,
                                           ascending=self.ascending,
                                           t_start=self.t_start,
                                           seed=int(self.rng.randint(2**32)))
        self.bg_streams = [self.bg_x]
        
        if self.num_pols == 2:
            self.bg_y = data_stream.DataStream(sample_rate=self.sample_rate,
                                               fch1=self.fch1,
                                               ascending=self.ascending,
                                               t_start=self.t_start,
                                               seed=int(self.rng.randint(2**32)))
            self.bg_streams.append(self.bg_y)
        
        self.antennas = []
        for i in range(self.num_antennas):
            antenna = Antenna(sample_rate=self.sample_rate,
                              fch1=self.fch1,
                              ascending=self.ascending,
                              num_pols=self.num_pols,
                              t_start=self.t_start,
                              seed=int(self.rng.randint(2**32)),
                              delay=delays[i])
            self.antennas.append(antenna)
            
    def set_time(self, t):
        self.start_obs = True
        self.t_start = t
        self.bg_x.set_time(t)
        if self.num_pols == 2:
            self.bg_y.set_time(t)
        for antenna in self.antennas:
            antenna.bg_cache = [None, None]
            antenna.set_time(t)
        
    def add_time(self, t):
        self.set_time(self.t_start + t)
        
    def reset_start(self):
        self.add_time(0)
            
    def get_samples(self, num_samples, total_obs_num_samples=None):
        # Check that num_samples is always larger than the maximum antenna delay
        assert num_samples > self.max_delay
        
        if self.start_obs:
            bg_num_samples = num_samples + self.max_delay
        else:
            bg_num_samples = num_samples
        self.bg_x.get_samples(bg_num_samples, total_obs_num_samples=total_obs_num_samples)
        if self.num_pols == 2:
            self.bg_y.get_samples(bg_num_samples, total_obs_num_samples=total_obs_num_samples)
        
        # For each antenna, get samples from each pol data stream, adding background contributions 
        # and caching voltages to account for varying antenna delays
        for antenna in self.antennas:
            antenna.x.bg_noise_std = self.bg_x.noise_std
            antenna.x.get_samples(num_samples, total_obs_num_samples=total_obs_num_samples)
            
            if self.start_obs:
                bg_x_v = self.bg_x.v[self.max_delay-antenna.delay:bg_num_samples-antenna.delay]
            else:
                bg_x_v = xp.concatenate([antenna.bg_cache[0], self.bg_x.v])[:bg_num_samples]
                
            antenna.bg_cache[0] = self.bg_x.v[bg_num_samples-antenna.delay:]
            antenna.x.v += bg_x_v
            
            if self.num_pols == 2:
                antenna.y.bg_noise_std = self.bg_y.noise_std
                antenna.y.get_samples(num_samples, total_obs_num_samples=total_obs_num_samples)
                
                if self.start_obs:
                    bg_y_v = self.bg_y.v[self.max_delay-antenna.delay:bg_num_samples-antenna.delay]
                else:
                    bg_y_v = xp.concatenate([antenna.bg_cache[1], self.bg_y.v])[:bg_num_samples]
                    
                antenna.bg_cache[1] = self.bg_y.v[bg_num_samples-antenna.delay:]
                antenna.y.v += bg_y_v
                
        self.t_start += num_samples * self.dt
        self.start_obs = False
        
        if self.num_pols == 2:
            return [[antenna.x.v, antenna.y.v] for antenna in self.antennas]
        else:
            return [[antenna.x.v] for antenna in self.antennas]
            
        