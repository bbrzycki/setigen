try:
    import cupy as xp
except ImportError:
    import numpy as xp

from astropy import units as u

from setigen import unit_utils
from . import data_stream


class Antenna(object):
    def __init__(self,
                 sample_rate,
                 num_pols=2,
                 seed=None,
                 delay=0):
        self.rng = xp.random.RandomState(seed)
        self.delay = delay
        
        self.sample_rate = unit_utils.get_value(sample_rate, u.Hz)
        assert num_pols in [1, 2]
        self.num_pols = num_pols
        self.start_obs = True
        
        self.x = data_stream.DataStream(sample_rate,
                                        int(self.rng.randint(2**32)))
        
        if self.num_pols == 2:
            self.y = data_stream.DataStream(sample_rate,
                                            int(self.rng.randint(2**32)))
        
        self.bg_cache = [None, None]
        
    def get_samples(self, num_samples):
        if self.num_pols == 2:
            samples = [[self.x.get_samples(num_samples), self.y.get_samples(num_samples)]]
        else:
            samples = [[self.x.get_samples(num_samples)]]
        self.start_obs = False
        return samples

        
class MultiAntennaArray(object):
    def __init__(self,
                 num_antennas,
                 sample_rate,
                 num_pols=2,
                 delays=None,
                 seed=None):
        self.rng = xp.random.RandomState(seed)
        if delays is None:
            self.delays = xp.zeros(num_antennas)
        else:
            assert len(delays) == num_antennas
            self.delays = xp.array(delays)
        self.max_delay = int(xp.max(delays))
        
        self.num_antennas = num_antennas
        self.sample_rate = unit_utils.get_value(sample_rate, u.Hz)
        assert num_pols in [1, 2]
        self.num_pols = num_pols
        self.start_obs = True
        
        self.x_bg = data_stream.DataStream(sample_rate,
                                           int(self.rng.randint(2**32)))
        if self.num_pols == 2:
            self.y_bg = data_stream.DataStream(sample_rate,
                                               int(self.rng.randint(2**32)))
        
        self.antennas = []
        for i in range(self.num_antennas):
            antenna = Antenna(sample_rate,
                              num_pols,
                              int(self.rng.randint(2**32)),
                              delay=delays[i])
            self.antennas.append(antenna)
            
    def get_samples(self, num_samples):
        # Check that num_samples is always larger than the maximum antenna delay
        assert num_samples > self.max_delay
        
        if self.start_obs:
            bg_num_samples = num_samples + self.max_delay
        else:
            bg_num_samples = num_samples
        self.x_bg.get_samples(bg_num_samples)
        if self.num_pols == 2:
            self.y_bg.get_samples(bg_num_samples)
        
        for antenna in self.antennas:
            antenna.x.get_samples(num_samples)
            if self.start_obs:
                bg_x = self.x_bg.v[self.max_delay-antenna.delay:bg_num_samples-antenna.delay]
            else:
                bg_x = xp.concatenate([antenna.bg_cache[0], self.x_bg.v])[:bg_num_samples]
            antenna.bg_cache[0] = self.x_bg.v[bg_num_samples-antenna.delay:]
            antenna.x.v += bg_x
            
            if self.num_pols == 2:
                antenna.y.get_samples(num_samples)
                if self.start_obs:
                    bg_y = self.y_bg.v[self.max_delay-antenna.delay:bg_num_samples-antenna.delay]
                else:
                    bg_y = xp.concatenate([antenna.bg_cache[1], self.y_bg.v])[:bg_num_samples]
                antenna.bg_cache[1] = self.y_bg.v[bg_num_samples-antenna.delay:]
                antenna.y.v += bg_y
                
        self.start_obs = False
        if self.num_pols == 2:
            return [[antenna.x.v, antenna.y.v] for antenna in self.antennas]
        else:
            return [[antenna.x.v] for antenna in self.antennas]
            
        