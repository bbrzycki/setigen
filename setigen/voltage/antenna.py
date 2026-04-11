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

from ._antenna.array_ops import (
    _apply_background_to_antenna,
    _collect_array_samples,
    _coerce_delays,
    _get_single_antenna_samples,
    _populate_background_streams,
    _reset_array_time_state,
    _set_antenna_time,
)
from ._antenna.construction import (
    _build_antennas,
    _build_background_streams,
    _build_polarization_streams,
    _coerce_fch1,
    _coerce_sample_rate,
    _validate_num_pols,
)


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
            Central frequency of the first coarse channel, in Hz.
            If ``ascending=True``, ``fch1`` is the minimum frequency; if ``ascending=False`` 
            (default), ``fch1`` is the maximum frequency.
        ascending : bool, optional
            Specify whether frequencies should be in ascending or descending order. Default 
            is True, for which ``fch1`` is the minimum frequency.
        num_pols : int, optional
            Number of polarizations, can be 1 or 2
        t_start : float, optional
            Start time, in seconds
        seed : None, int, Generator, optional
            Random seed or seed generator
        """
        self.rng = xp.random.default_rng(seed)
        
        self.sample_rate = _coerce_sample_rate(sample_rate)
        self.dt = 1 / self.sample_rate
        
        self.fch1 = _coerce_fch1(fch1)
        self.ascending = ascending
        
        self.num_pols = _validate_num_pols(num_pols)
        
        self.t_start = t_start
        self.start_obs = True
        
        self.x, self.y, self.streams = _build_polarization_streams(sample_rate=self.sample_rate,
                                                                   fch1=self.fch1,
                                                                   ascending=self.ascending,
                                                                   t_start=self.t_start,
                                                                   num_pols=self.num_pols,
                                                                   rng=self.rng)
        
        self.delay = None
        self.bg_cache = [None, None]
        
    def set_time(self, t):
        """
        Set start time before next set of samples.
        """
        _set_antenna_time(self, t)
        
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
        return _get_single_antenna_samples(self, num_samples, xp=xp)

        
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
            Central frequency of the first coarse channel, in Hz.
            If ``ascending=True``, ``fch1`` is the minimum frequency; if ``ascending=False`` 
            (default), ``fch1`` is the maximum frequency.
        ascending : bool, optional
            Specify whether frequencies should be in ascending or descending order. Default 
            is True, for which ``fch1`` is the minimum frequency.
        num_pols : int, optional
            Number of polarizations, can be 1 or 2
        delays : array, optional
            Array of integers specifying relative delay offsets per array with respect to the coherent antenna 
            array background. If None, uses 0 delay for all Antennas.
        t_start : float, optional
            Start time, in seconds
        seed : None, int, Generator, optional
            Random seed or seed generator
        """
        self.rng = xp.random.default_rng(seed)
        
        self.delays, self.max_delay = _coerce_delays(delays=delays,
                                                     num_antennas=num_antennas,
                                                     xp=xp)
        
        self.num_antennas = num_antennas
        self.sample_rate = _coerce_sample_rate(sample_rate)
        self.dt = 1 / self.sample_rate
        
        self.fch1 = _coerce_fch1(fch1)
        self.ascending = ascending
        
        self.num_pols = _validate_num_pols(num_pols)
        
        self.t_start = t_start
        self.start_obs = True
        
        self.antennas = _build_antennas(num_antennas=self.num_antennas,
                                        sample_rate=self.sample_rate,
                                        fch1=self.fch1,
                                        ascending=self.ascending,
                                        num_pols=self.num_pols,
                                        t_start=self.t_start,
                                        rng=self.rng,
                                        antenna_cls=Antenna,
                                        delays=self.delays)
        
        # Create background data streams and link relevant antenna data streams for tracking noise
        self.bg_x, self.bg_y, self.bg_streams = _build_background_streams(sample_rate=self.sample_rate,
                                                                          fch1=self.fch1,
                                                                          ascending=self.ascending,
                                                                          t_start=self.t_start,
                                                                          num_pols=self.num_pols,
                                                                          rng=self.rng,
                                                                          antennas=self.antennas)
            
    def set_time(self, t):
        """
        Set start time before next set of samples.
        """
        _reset_array_time_state(self, t)
        
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
        with the Antenna, according to the delay, so that regardless of ``num_samples``, each Antenna 
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
        if num_samples <= self.max_delay:
            raise ValueError("num_samples must be greater than the maximum antenna delay")

        bg_num_samples = _populate_background_streams(self, num_samples)

        # For each antenna, get samples from each pol data stream, adding background contributions
        # and caching voltages to account for varying antenna delays
        for antenna in self.antennas:
            _apply_background_to_antenna(self,
                                         antenna,
                                         bg_num_samples=bg_num_samples,
                                         num_samples=num_samples,
                                         xp=xp)
                
        self.t_start += num_samples * self.dt
        self.start_obs = False

        return _collect_array_samples(self, xp=xp)
            
        
