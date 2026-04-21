from __future__ import annotations

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
from typing import Any

from ._antenna.array_ops import (
    _apply_background_to_antenna,
    _collect_array_samples,
    _coerce_delays,
    _get_single_antenna_samples,
    _populate_background_streams,
    _reset_array_time_state,
    _set_antenna_time,
)
from setigen._typing import SeedLike
from ._antenna.construction import (
    _build_antennas,
    _build_background_streams,
    _build_polarization_streams,
    _coerce_fch1,
    _coerce_sample_rate,
    _validate_num_pols,
)


class Antenna(object):
    """Model a radio antenna with one or two polarization streams."""
    def __init__(self,
                 sample_rate: float | u.Quantity = 3*u.GHz,
                 fch1: float | u.Quantity = 0*u.GHz,
                 ascending: bool = True,
                 num_pols: int = 2,
                 t_start: float = 0,
                 seed: SeedLike = None,
                 **kwargs: Any) -> None:
        """Initialize an antenna and its polarization data streams.

        Args:
            sample_rate: Real-voltage sample rate.
            fch1: Frequency of the first coarse channel.
            ascending: Whether the frequency axis is ascending.
            num_pols: Number of polarizations, either one or two.
            t_start: Start time in seconds.
            seed: Random seed or generator.
            **kwargs: Reserved keyword arguments.
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
        
    def set_time(self, t: float) -> None:
        """Set the antenna start time for the next sample request.

        Args:
            t: New start time in seconds.
        """
        _set_antenna_time(self, t)
        
    def add_time(self, t: float) -> None:
        """Advance the antenna start time.

        Args:
            t: Time increment in seconds.
        """
        self.set_time(self.t_start + t)
        
    def reset_start(self) -> None:
        """Reset the observation-start state for the antenna."""
        self.add_time(0)
        
    def get_samples(self, num_samples: int) -> np.ndarray:
        """Retrieve voltage samples from each polarization.

        Args:
            num_samples: Number of samples to retrieve.

        Returns:
            Voltage samples of shape `(1, num_pols, num_samples)`.
        """
        return _get_single_antenna_samples(self, num_samples, xp=xp)

        
class MultiAntennaArray(object):
    """Model an antenna array with per-antenna delays and shared background noise."""
    def __init__(self,
                 num_antennas: int,
                 sample_rate: float | u.Quantity = 3*u.GHz,
                 fch1: float | u.Quantity = 0*u.GHz,
                 ascending: bool = True,
                 num_pols: int = 2,
                 delays: list[int] | tuple[int, ...] | None = None,
                 t_start: float = 0,
                 seed: SeedLike = None,
                 **kwargs: Any) -> None:
        """Initialize a multi-antenna array.

        Args:
            num_antennas: Number of antennas in the array.
            sample_rate: Real-voltage sample rate.
            fch1: Frequency of the first coarse channel.
            ascending: Whether the frequency axis is ascending.
            num_pols: Number of polarizations, either one or two.
            delays: Optional per-antenna integer-sample delays.
            t_start: Start time in seconds.
            seed: Random seed or generator.
            **kwargs: Reserved keyword arguments.
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
            
    def set_time(self, t: float) -> None:
        """Set the array start time for the next sample request.

        Args:
            t: New start time in seconds.
        """
        _reset_array_time_state(self, t)
        
    def add_time(self, t: float) -> None:
        """Advance the array start time.

        Args:
            t: Time increment in seconds.
        """
        self.set_time(self.t_start + t)
        
    def reset_start(self) -> None:
        """Reset the observation-start state for the array."""
        self.add_time(0)
            
    def get_samples(self, num_samples: int) -> xp.ndarray:
        """Retrieve voltage samples from each antenna and polarization.

        Args:
            num_samples: Number of samples to retrieve.

        Returns:
            Voltage samples of shape `(num_antennas, num_pols, num_samples)`.

        Raises:
            ValueError: If `num_samples` does not exceed the maximum antenna
                delay.
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
            
        
