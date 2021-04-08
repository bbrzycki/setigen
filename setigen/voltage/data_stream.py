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
    Facilitate noise and signal injection in a real voltage time series data stream,
    for a single polarization. Noise and signal sources are functions, saved as 
    properties of the DataStream, so that individual samples can be queried using
    get_samples(). 
    """
    
    def __init__(self,
                 sample_rate=3*u.GHz,
                 fch1=0*u.GHz,
                 ascending=True,
                 t_start=0,
                 seed=None):
        """
        Initialize a DataStream object with a sampling rate and frequency range.

        By default, :code:`setigen.voltage` does not employ heterodyne mixing and filtering
        to focus on a frequency bandwidth. Instead, the sensitive range is determined
        by these parameters; starting at the frequency `fch1` and spanning the Nyquist 
        range `sample_rate / 2` in the increasing or decreasing frequency direction,
        as specified by `ascending`. Note that accordingly, the spectral response will
        be susceptible to aliasing, so take care that the desired frequency range is
        correct and that signals are injected at appropriate frequencies. 

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
        t_start : float, optional
            Start time, in seconds
        seed : int, optional
            Integer seed between 0 and 2**32. If None, the random number generator
            will use a random seed.
        """
        #: Random number generator
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
        """
        Set array of times for voltage calculation, and reset voltage array.
        """
        self.ts = self.t_start + xp.linspace(0., 
                                             num_samples * self.dt,
                                             num_samples,
                                             endpoint=False)
        self.t_start += num_samples * self.dt
        self.v = xp.zeros(num_samples)
        
    def set_time(self, t):
        """
        Set start time before next set of samples.
        """
        self.start_obs = True
        self.t_start = t
        
    def add_time(self, t):
        """
        Add time before next set of samples.
        """
        self.set_time(self.t_start + t)
        
    def update_noise(self, stats_calc_num_samples=10000):
        """
        Replace self.noise_std by calculating out a few samples and estimating the 
        standard deviation of the voltages.

        Parameters
        ----------
        stats_calc_num_samples : int, optional
            Maximum number of samples for use in estimating noise standard deviation
        """
        start_obs = self.start_obs
        t_start = self.t_start
        
        v = self.get_samples(num_samples=stats_calc_num_samples)
        _, self.noise_std = estimate_stats(v, stats_calc_num_samples=stats_calc_num_samples)
        
        self.start_obs = start_obs
        self.t_start = t_start
        
    def get_total_noise_std(self):
        """
        Get the standard deviation of the noise. If this DataStream is part
        of an array of Antennas, this will account for the background noise in the 
        corresponding polarization.
        
        Note that if this DataStream has custom signals or noise, it might not
        'know' what the noise standard deviation is. In this case, one should run
        :func:`~setigen.voltage.data_stream.DataStream.update_noise()` to update the 
        DataStream's estimate for the noise. Note that this actually runs 
        :func:`~setigen.voltage.data_stream.DataStream.get_samples()` for the calculation, so
        if your custom signal functions have mutable properties, make sure to reset these
        (if necessary) before saving out data. 
        """
        return xp.sqrt(self.noise_std**2 + self.bg_noise_std**2)
    
    def add_noise(self, v_mean, v_std):
        """
        Add Gaussian noise source to data stream. This essentially adds a lambda function that
        gets the appropriate number of noise samples to add to the voltage array when 
        :code:`get_samples()` is called. Updates noise property to reflect
        added noise.
        
        Parameters
        ----------
        v_mean : float
            Noise mean
        v_std : float
            Noise standard deviation
        """
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
        Adds a drifting cosine signal (linear chirp) as a signal source function. 
        
        Parameters
        ----------
        f_start : float
            Starting signal frequency
        drift_rate : float
            Drift rate in Hz / s
        level : float
            Signal level or amplitude
        phase : float
            Phase, in radiations
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
        """
        Wrapper function to add a custom signal source function.
        """
        self.signal_sources.append(signal_func)
    
    def get_samples(self, num_samples):
        """
        Retrieve voltage samples, based on noise and signal source functions.
        
        If custom signals add complex voltages, the voltage array will be cast to
        complex type.
        
        Parameters
        ----------
        num_samples : int
            Number of samples to get
            
        Returns
        -------
        v : array
            Array of voltage samples
        """
        self._update_t(num_samples)
        
        for noise_func in self.noise_sources:
            self.v += noise_func(self.ts)
            
        for signal_func in self.signal_sources:
            # Ensure that the array is of the correct type
            signal_v = xp.array(signal_func(self.ts))
            # If there are complex voltages, make sure to cast self.v to complex
            if not xp.iscomplexobj(self.v) and xp.iscomplexobj(signal_v):
                self.v = self.v.astype(complex)
            self.v += signal_v
            
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
        """
        Initialize a BackgroundDataStream object with a sampling rate and frequency range.
        The main extension is that we also pass in a list of DataStreams, belonging to all
        the Antennas within a MultiAntennaArray, for the same corresponding polarization. 
        When noise is added to a BackgroundDataStream, the noise standard deviation gets 
        propagated to each Antenna DataStream via the :code:`DataStream.bg_noise_std` property.

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
        t_start : float, optional
            Start time, in seconds
        seed : int, optional
            Integer seed between 0 and 2**32. If None, the random number generator
            will use a random seed.
        antenna_streams : list of DataStream objects
            List of DataStreams, which belong to the Antennas in a MultiAntennaArray,
            all corresponding to the same polarization
        """
        super().__init__(sample_rate=sample_rate,
                         fch1=fch1,
                         ascending=ascending,
                         t_start=t_start,
                         seed=seed)
        self.antenna_streams = antenna_streams
        
    def _set_all_bg_noise(self):
        """
        Helper function to set background noise standard deviation of each
        antenna for the corresponding polarization equal to that of this stream.
        """
        for stream in self.antenna_streams:
            stream.bg_noise_std = self.noise_std
        
    def update_noise(self, stats_calc_num_samples=10000):
        """
        Replace self.noise_std by calculating out a few samples and estimating the 
        standard deviation of the voltages. Further, set all child antenna background
        noise values.

        Parameters
        ----------
        stats_calc_num_samples : int, optional
            Maximum number of samples for use in estimating noise standard deviation
        """
        DataStream.update_noise(self, stats_calc_num_samples=stats_calc_num_samples)
        self._set_all_bg_noise()
        
    def add_noise(self, v_mean, v_std):
        """
        Add Gaussian noise source to data stream. This essentially adds a lambda function that
        gets the appropriate number of noise samples to add to the voltage array when 
        :code:`get_samples()` is called. Updates noise property to reflect
        added noise. Further, set all child antenna background noise values.
        
        Parameters
        ----------
        v_mean : float
            Noise mean
        v_std : float
            Noise standard deviation
        """
        DataStream.add_noise(self, v_mean, v_std)
        self._set_all_bg_noise()
        
        
def estimate_stats(voltages, stats_calc_num_samples=10000):
    """
    Estimate mean and standard deviation, truncating to at most `stats_calc_num_samples` samples 
    to reduce computation.

    Parameters
    ----------
    voltages : array
        Array of voltages
    stats_calc_num_samples : int, optional
        Maximum number of samples for use in estimating noise statistics
        
    Returns
    -------
    data_mean : float
        Mean of voltages
    data_sigma : float
        Standard deviation of voltages
    """
    calc_len = xp.amin(xp.array([stats_calc_num_samples, len(voltages)]))
    data_sigma = xp.std(voltages[:calc_len])
    data_mean = xp.mean(voltages[:calc_len])
    
    return data_mean, data_sigma