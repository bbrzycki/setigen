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

from setigen import unit_utils
from . import data_stream


class Antenna(object):
    """
    Models a radio antenna, with a DataStream per polarization (one or two). 
    """
    def __init__(self,
                 sample_rate=3*u.GHz,
                 fch1=0*u.GHz,
                 ascending=True,
                 num_pols=2,
                 t_start=0,
                 seed=None,
                 **kwargs):
        """
        Initialize an Antenna object, which creates DataStreams for each polarization, under
        Antenna.x and Antenna.y (if there is a second polarization).

        Parameters
        ----------
        sample_rate : float, optional
            Physical sample rate, in Hz, for collecting real voltage data
        fch1 : astropy.Quantity, optional
            Starting frequency of the first coarse channel, in Hz.
            If ascending=True, fch1 is the minimum frequency; if ascending=False 
            (default), fch1 is the maximum frequency.
        ascending : bool, optional
            Specify whether frequencies should be in ascending or descending order. Default 
            is True, for which fch1 is the minimum frequency.
        num_pols : int, optional
            Number of polarizations, can be 1 or 2
        t_start : float, optional
            Start time, in seconds
        seed : int, optional
            Integer seed between 0 and 2**32. If None, the random number generator
            will use a random seed.
        """
        self.rng = xp.random.RandomState(seed)
        
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
        
        self.delay = None
        self.bg_cache = [None, None]
        
    def set_time(self, t):
        """
        Set start time before next set of samples.
        """
        self.start_obs = True
        self.t_start = t
        self.x.set_time(t)
        if self.num_pols == 2:
            self.y.set_time(t)
        
    def add_time(self, t):
        """
        Add time before next set of samples.
        """
        self.set_time(self.t_start + t)
        
    def reset_start(self):
        """
        Reset the boolean that tracks whether this is the start of an observation.
        """
        self.add_time(0)
        
    def get_samples(self, num_samples):
        """
        Retrieve voltage samples from each polarization.
        
        Parameters
        ----------
        num_samples : int
            Number of samples to get
            
        Returns
        -------
        samples : array
            Array of voltage samples, of shape (1, num_pols, num_samples)
        """
        if self.num_pols == 2:
            samples = [[self.x.get_samples(num_samples), 
                        self.y.get_samples(num_samples)]]
        else:
            samples = [[self.x.get_samples(num_samples)]]
            
        self.t_start += num_samples * self.dt
        self.start_obs = False
        
        return xp.array(samples)

        
class MultiAntennaArray(object):
    """
    Models a radio antenna array, with list of Antennas, subject to user-specified sample delays.
    """
    def __init__(self,
                 num_antennas,
                 sample_rate=3*u.GHz,
                 fch1=0*u.GHz,
                 ascending=True,
                 num_pols=2,
                 delays=None,
                 t_start=0,
                 seed=None,
                 **kwargs):
        """
        Initialize a MultiAntennaArray object, which creates a list of Antenna objects, each with a specified
        relative integer sample delay. Also creates background DataStreams to model coherent noise present in 
        each Antenna, subject to that Antenna's delay. 

        Parameters
        ----------
        num_antennas : int
            Number of Antennas in the array
        sample_rate : float, optional
            Physical sample rate, in Hz, for collecting real voltage data
        fch1 : astropy.Quantity, optional
            Starting frequency of the first coarse channel, in Hz.
            If ascending=True, fch1 is the minimum frequency; if ascending=False 
            (default), fch1 is the maximum frequency.
        ascending : bool, optional
            Specify whether frequencies should be in ascending or descending order. Default 
            is True, for which fch1 is the minimum frequency.
        num_pols : int, optional
            Number of polarizations, can be 1 or 2
        delays : array, optional
            Array of integers specifying relative delay offsets per array with respect to the coherent antenna 
            array background. If None, uses 0 delay for all Antennas.
        t_start : float, optional
            Start time, in seconds
        seed : int, optional
            Integer seed between 0 and 2**32. If None, the random number generator
            will use a random seed.
        """
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
        
        self.antennas = []
        for i in range(self.num_antennas):
            antenna = Antenna(sample_rate=self.sample_rate,
                              fch1=self.fch1,
                              ascending=self.ascending,
                              num_pols=self.num_pols,
                              t_start=self.t_start,
                              seed=int(self.rng.randint(2**32)))
            antenna.delay = delays[i]
            self.antennas.append(antenna)
        
        # Create background data streams and link relevant antenna data streams for tracking noise
        self.bg_x = data_stream.BackgroundDataStream(sample_rate=self.sample_rate,
                                                     fch1=self.fch1,
                                                     ascending=self.ascending,
                                                     t_start=self.t_start,
                                                     seed=int(self.rng.randint(2**32)),
                                                     antenna_streams=[antenna.x for antenna in self.antennas])
        self.bg_streams = [self.bg_x]
        
        if self.num_pols == 2:
            self.bg_y = data_stream.BackgroundDataStream(sample_rate=self.sample_rate,
                                                         fch1=self.fch1,
                                                         ascending=self.ascending,
                                                         t_start=self.t_start,
                                                         seed=int(self.rng.randint(2**32)),
                                                         antenna_streams=[antenna.y for antenna in self.antennas])
            self.bg_streams.append(self.bg_y)
            
    def set_time(self, t):
        """
        Set start time before next set of samples.
        """
        self.start_obs = True
        self.t_start = t
        self.bg_x.set_time(t)
        if self.num_pols == 2:
            self.bg_y.set_time(t)
        for antenna in self.antennas:
            antenna.bg_cache = [None, None]
            antenna.set_time(t)
        
    def add_time(self, t):
        """
        Add time before next set of samples.
        """
        self.set_time(self.t_start + t)
        
    def reset_start(self):
        """
        Reset the boolean that tracks whether this is the start of an observation.
        """
        self.add_time(0)
            
    def get_samples(self, num_samples):
        """
        Retrieve voltage samples from each antenna and polarization.
        
        First, background data stream voltages are computed. Then, for each Antenna, voltages
        are retrieved per polarization and summed with the corresponding background voltages, subject
        to that Antenna's sample delay. An appropriate number of background voltage samples are cached 
        with the Antenna, according to the delay, so that regardless of `num_samples`, each Antenna 
        data stream has enough background samples to add.
        
        Parameters
        ----------
        num_samples : int
            Number of samples to get
            
        Returns
        -------
        samples : array
            Array of voltage samples, of shape (num_antennas, num_pols, num_samples)
        """
        # Check that num_samples is always larger than the maximum antenna delay
        assert num_samples > self.max_delay
        
        if self.start_obs:
            bg_num_samples = num_samples + self.max_delay
        else:
            bg_num_samples = num_samples
        self.bg_x.get_samples(bg_num_samples)
        if self.num_pols == 2:
            self.bg_y.get_samples(bg_num_samples)
        
        # For each antenna, get samples from each pol data stream, adding background contributions 
        # and caching voltages to account for varying antenna delays
        for antenna in self.antennas:
            antenna.x.get_samples(num_samples)
            
            if self.start_obs:
                bg_x_v = self.bg_x.v[self.max_delay-antenna.delay:bg_num_samples-antenna.delay]
            else:
                bg_x_v = xp.concatenate([antenna.bg_cache[0], self.bg_x.v])[:bg_num_samples]
                
            antenna.bg_cache[0] = self.bg_x.v[bg_num_samples-antenna.delay:]
            antenna.x.v += bg_x_v
            
            if self.num_pols == 2:
                antenna.y.get_samples(num_samples)
                
                if self.start_obs:
                    bg_y_v = self.bg_y.v[self.max_delay-antenna.delay:bg_num_samples-antenna.delay]
                else:
                    bg_y_v = xp.concatenate([antenna.bg_cache[1], self.bg_y.v])[:bg_num_samples]
                    
                antenna.bg_cache[1] = self.bg_y.v[bg_num_samples-antenna.delay:]
                antenna.y.v += bg_y_v
                
        self.t_start += num_samples * self.dt
        self.start_obs = False
        
        if self.num_pols == 2:
            samples = [[antenna.x.v, antenna.y.v] for antenna in self.antennas]
        else:
            samples = [[antenna.x.v] for antenna in self.antennas]
        return xp.array(samples)
            
        