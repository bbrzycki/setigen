from __future__ import annotations

from collections.abc import Callable

from astropy import units as u
from setigen import unit_utils
from setigen._typing import SeedLike
from ._array_backend import xp


class DataStream(object):
    """Model a single-polarization real-voltage data stream."""
    
    def __init__(self,
                 sample_rate: float | u.Quantity = 3*u.GHz,
                 fch1: float | u.Quantity = 0*u.GHz,
                 ascending: bool = True,
                 t_start: float = 0,
                 seed: SeedLike = None) -> None:
        """Initialize a real-voltage data stream.

        Args:
            sample_rate: Real-voltage sample rate.
            fch1: Frequency of the first coarse channel.
            ascending: Whether the frequency axis is ascending.
            t_start: Start time in seconds.
            seed: Random seed or generator.
        """
        #: Random number generator
        self.rng = xp.random.default_rng(seed)

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
        self.noise_sources: list[Callable[[xp.ndarray], xp.ndarray]] = []
        self.signal_sources: list[Callable[[xp.ndarray], xp.ndarray]] = []
        
    def _update_t(self, num_samples: int) -> None:
        """Update the current sample times and reset the voltage buffer.

        Args:
            num_samples: Number of samples to generate next.
        """
        self.ts = self.t_start + xp.linspace(0.,
                                             num_samples * self.dt,
                                             num_samples,
                                             endpoint=False)
        self.t_start += num_samples * self.dt
        self.v = xp.zeros(num_samples)
        
    def set_time(self, t: float) -> None:
        """Set the start time for the next sample request.

        Args:
            t: New start time in seconds.
        """
        self.start_obs = True
        self.t_start = t
        
    def add_time(self, t: float) -> None:
        """Advance the start time for the next sample request.

        Args:
            t: Time increment in seconds.
        """
        self.set_time(self.t_start + t)
        
    def update_noise(self, stats_calc_num_samples: int = 10000) -> None:
        """Estimate and update the stream noise standard deviation.

        Args:
            stats_calc_num_samples: Maximum number of samples to use.
        """
        start_obs = self.start_obs
        t_start = self.t_start

        voltages = self.get_samples(num_samples=stats_calc_num_samples)
        _, self.noise_std = estimate_stats(voltages,
                                           stats_calc_num_samples=stats_calc_num_samples)

        self.start_obs = start_obs
        self.t_start = t_start
        
    def get_total_noise_std(self) -> float:
        """Return the combined intrinsic and background noise standard deviation.

        Returns:
            Total noise standard deviation including background noise.
        """
        return xp.sqrt(self.noise_std**2 + self.bg_noise_std**2)
    
    def add_noise(self, v_mean: float, v_std: float) -> None:
        """Add a Gaussian noise source to the stream.

        Args:
            v_mean: Noise mean.
            v_std: Noise standard deviation.
        """
        noise_func = lambda ts: v_mean + v_std * self.rng.standard_normal(size=len(ts))

        # Variances add, not standard deviations
        self.noise_std = xp.sqrt(self.noise_std**2 + v_std**2)
        self.noise_sources.append(noise_func)
        
    def add_constant_signal(self,
                            f_start: float,
                            drift_rate: float,
                            level: float,
                            phase: float=0) -> None:
        """Add a drifting cosine signal to the stream.

        Args:
            f_start: Starting signal frequency.
            drift_rate: Drift rate in Hz/s.
            level: Signal amplitude.
            phase: Signal phase in radians.
        """
        f_start = unit_utils.get_value(f_start, u.Hz)
        drift_rate = unit_utils.get_value(drift_rate, u.Hz / u.s)

        def signal_func(ts: xp.ndarray) -> xp.ndarray:
            # Calculate adjusted center frequencies, according to chirp
            chirp_phase = 2 * xp.pi * ((f_start - self.fch1) * ts + 0.5 * drift_rate * ts**2)
            if not self.ascending:
                chirp_phase = -chirp_phase
            return level * xp.cos(chirp_phase + phase)

        self.signal_sources.append(signal_func)
        
    def add_signal(self, signal_func: Callable[[xp.ndarray], xp.ndarray]) -> None:
        """Add a custom signal source function to the stream.

        Args:
            signal_func: Callable that maps time samples to voltages.
        """
        self.signal_sources.append(signal_func)
    
    def get_samples(self, num_samples: int) -> xp.ndarray:
        """Generate voltage samples from the configured sources.

        Args:
            num_samples: Number of samples to generate.

        Returns:
            Array of generated voltage samples.
        """
        self._update_t(num_samples)

        for noise_func in self.noise_sources:
            self.v += noise_func(self.ts)

        for signal_func in self.signal_sources:
            signal_v = xp.array(signal_func(self.ts))
            if not xp.iscomplexobj(self.v) and xp.iscomplexobj(signal_v):
                self.v = self.v.astype(complex)
            self.v += signal_v

        self.start_obs = False

        return self.v
        
        
class BackgroundDataStream(DataStream):
    """Data stream used for coherent background noise in antenna arrays."""
    
    def __init__(self,
                 sample_rate: float | u.Quantity = 3*u.GHz,
                 fch1: float | u.Quantity = 0*u.GHz,
                 ascending: bool = True,
                 t_start: float = 0,
                 seed: SeedLike = None,
                 antenna_streams: list[DataStream] | tuple[DataStream, ...] | None = None) -> None:
        """Initialize a background data stream.

        Args:
            sample_rate: Real-voltage sample rate.
            fch1: Frequency of the first coarse channel.
            ascending: Whether the frequency axis is ascending.
            t_start: Start time in seconds.
            seed: Random seed or generator.
            antenna_streams: Antenna streams that share this background noise.
        """
        super().__init__(sample_rate=sample_rate,
                         fch1=fch1,
                         ascending=ascending,
                         t_start=t_start,
                         seed=seed)
        self.antenna_streams = [] if antenna_streams is None else list(antenna_streams)
        
    def _set_all_bg_noise(self) -> None:
        """Propagate background noise statistics to child antenna streams."""
        for stream in self.antenna_streams:
            stream.bg_noise_std = self.noise_std
        
    def update_noise(self, stats_calc_num_samples: int = 10000) -> None:
        """Estimate background noise and propagate it to antenna streams.

        Args:
            stats_calc_num_samples: Maximum number of samples to use.
        """
        DataStream.update_noise(self, stats_calc_num_samples=stats_calc_num_samples)
        self._set_all_bg_noise()
        
    def add_noise(self, v_mean: float, v_std: float) -> None:
        """Add Gaussian background noise and propagate its variance.

        Args:
            v_mean: Noise mean.
            v_std: Noise standard deviation.
        """
        DataStream.add_noise(self, v_mean, v_std)
        self._set_all_bg_noise()
        
        
def estimate_stats(voltages: xp.ndarray,
                   stats_calc_num_samples: int = 10000) -> tuple[float, float]:
    """Estimate the mean and standard deviation of a voltage sequence.

    Args:
        voltages: Voltage array.
        stats_calc_num_samples: Maximum number of samples to use.

    Returns:
        Estimated mean and standard deviation.
    """
    calc_len = xp.amin(xp.array([stats_calc_num_samples, len(voltages)]))
    data_sigma = xp.std(voltages[:calc_len])
    data_mean = xp.mean(voltages[:calc_len])
    
    return data_mean, data_sigma
