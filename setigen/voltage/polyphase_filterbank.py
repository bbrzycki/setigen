from __future__ import annotations

import os
from typing import Literal

GPU_FLAG = os.getenv('SETIGEN_ENABLE_GPU', '0')
if GPU_FLAG == '1':
    try:
        import cupy as xp
    except ImportError:
        import numpy as xp
else:
    import numpy as xp
    
import scipy.signal

from setigen._typing import SeedLike


class PolyphaseFilterbank(object):
    """Implement a polyphase filterbank for coarse channelization."""
    
    def __init__(self, num_taps: int = 8, num_branches: int = 1024, window_fn: str = 'hamming') -> None:
        """Initialize a polyphase filterbank.

        Args:
            num_taps: Number of PFB taps.
            num_branches: Number of PFB branches.
            window_fn: Windowing function used for the PFB.
        """
        self.num_taps = num_taps
        self.num_branches = num_branches
        self.window_fn = window_fn
        
        self.cache = None
        
        self._get_pfb_window()
        
        # Estimate stds after channelizing Gaussian with mean 0, std 1
        self.channelized_stds = None
        
    def _reset_cache(self) -> None:
        """Clear the cached overlap samples."""
        self.cache = None
                
    def estimate_channelized_stds(self, factor: int = 10000, seed: SeedLike = None) -> xp.ndarray:
        """Estimate post-channelization real and imaginary standard deviations.

        Args:
            factor: Multiplier for `num_branches` used in the estimate.
            seed: Random seed or generator.

        Returns:
            Estimated real and imaginary standard deviations.
        """
        rng = xp.random.default_rng(seed)
        sample_v = rng.standard_normal(size=factor * self.num_branches)
        v_pfb = self.channelize(sample_v, cache=False)
        self.channelized_stds = xp.array([v_pfb.real.std(), v_pfb.imag.std()])
        return self.channelized_stds
        
    def _get_pfb_window(self) -> None:
        """Create and cache the PFB window coefficients."""
        self.window = get_pfb_window(self.num_taps, 
                                     self.num_branches, 
                                     self.window_fn)

    def get_response(self, fftlength: int = 512) -> xp.ndarray:
        """Return the half-coarse-channel PFB frequency response.

        Args:
            fftlength: Fine-channel FFT length used for the response estimate.

        Returns:
            Half-coarse-channel frequency response.

        Raises:
            ValueError: If `fftlength` is not a multiple of `num_taps`.
        """
        if fftlength % self.num_taps != 0:
            raise ValueError(f"fftlength ({fftlength}) must be a multiple of taps ({self.num_taps})")
        freq_response_x = xp.zeros(self.num_branches * fftlength)
        freq_response_x[:self.num_taps*self.num_branches] = self.window
        h = xp.fft.fft(freq_response_x)
        half_coarse_chan = (xp.abs(h)**2)[:fftlength//2]+(xp.abs(h)**2)[fftlength//2:fftlength][::-1]
        self.response = self.half_coarse_chan = half_coarse_chan
        self.max_mean_ratio = xp.max(half_coarse_chan) / xp.mean(half_coarse_chan)
        return half_coarse_chan

    def tile_response(self, num_chans: int, fftlength: int = 512) -> xp.ndarray:
        """Construct a tiled multi-channel PFB frequency response.

        Args:
            num_chans: Number of coarse channels to tile.
            fftlength: Fine-channel FFT length used for the response estimate.

        Returns:
            Tiled coarse-channel response.
        """
        response = self.get_response(fftlength=fftlength)
        return xp.tile(xp.concatenate([response[::-1], response]), 
                       num_chans)

    def channelize(self,
                   x: xp.ndarray,
                   cache: bool = True,
                   start_chan: int = 0,
                   num_chans: int | None = None,
                   method: Literal["auto", "full", "selected"] = "auto") -> xp.ndarray:
        """Channelize input voltages with the PFB and a normalized FFT.

        Args:
            x: Input voltage array.
            cache: Whether to retain overlap samples between calls.
            start_chan: First coarse channel to return.
            num_chans: Number of coarse channels to return. Defaults to all
                positive-frequency channels from ``start_chan`` onward.
            method: Coarse-channel transform method. ``"full"`` computes the
                full FFT then slices, ``"selected"`` computes only requested
                DFT bins, and ``"auto"`` selects the exact cheaper path for
                small channel selections.

        Returns:
            Post-FFT complex voltages.

        Raises:
            ValueError: If the selected channel range or method is invalid.
        """
        if method not in ("auto", "full", "selected"):
            raise ValueError("method must be one of 'auto', 'full', or 'selected'.")
        if start_chan < 0:
            raise ValueError("start_chan must be non-negative.")
        max_chans = self.num_branches // 2
        if num_chans is None:
            num_chans = max_chans - start_chan
        if num_chans <= 0 or start_chan + num_chans > max_chans:
            raise ValueError("Selected coarse channel range is out of bounds.")

        if cache:
            # Cache last section of data, which is excluded in PFB step
            if self.cache is not None:
                x = xp.concatenate([self.cache, x])
            self.cache = x[-self.num_taps*self.num_branches:]
        
        x = pfb_frontend(x, self.window, self.num_taps, self.num_branches)

        if method == "auto":
            selected_limit = min(8, max(1, self.num_branches // 64))
            method = "selected" if num_chans <= selected_limit else "full"

        if method == "selected":
            bins = xp.arange(start_chan, start_chan + num_chans)
            sample_idx = xp.arange(self.num_branches)
            twiddle = xp.exp(
                -2j * xp.pi * sample_idx[:, xp.newaxis] * bins[xp.newaxis, :] / self.num_branches
            )
            return (x @ twiddle) / self.num_branches**0.5

        X_pfb = xp.fft.fft(x,
                           self.num_branches,
                           axis=1)[:, 0:self.num_branches//2] / self.num_branches**0.5
        return X_pfb[:, start_chan:start_chan + num_chans]
    
    
def pfb_frontend_reference(x: xp.ndarray,
                           pfb_window: xp.ndarray,
                           num_taps: int,
                           num_branches: int) -> xp.ndarray:
    """Apply the polyphase frontend windowing operation.

    Args:
        x: Input voltage array.
        pfb_window: PFB window coefficients.
        num_taps: Number of PFB taps.
        num_branches: Number of PFB branches.

    Returns:
        Voltage array after PFB weighting.
    """
    W = int(len(x) / num_taps / num_branches)
    
    # Truncate data stream x to fit reshape step
    x_p = x[:W*num_taps*num_branches].reshape((W * num_taps, num_branches))
    h_p = pfb_window.reshape((num_taps, num_branches))
    
    # Resulting summed data array will be slightly shorter from windowing coeffs
    # I = xp.expand_dims(xp.arange(num_taps), 0) + xp.expand_dims(xp.arange((W - 1) * num_taps), 0).T
    # x_summed = xp.sum(x_p[I] * h_p, axis=1) / num_taps
    
    x_summed = xp.zeros(((W - 1) * num_taps, num_branches))
    for t in range(0, (W - 1) * num_taps):
        x_weighted = x_p[t:t+num_taps, :] * h_p
        x_summed[t, :] = xp.sum(x_weighted, axis=0)
    return x_summed


def pfb_frontend(x: xp.ndarray,
                 pfb_window: xp.ndarray,
                 num_taps: int,
                 num_branches: int) -> xp.ndarray:
    """Apply the vectorized polyphase frontend windowing operation.

    Args:
        x: Input voltage array.
        pfb_window: PFB window coefficients.
        num_taps: Number of PFB taps.
        num_branches: Number of PFB branches.

    Returns:
        Voltage array after PFB weighting.
    """
    W = int(len(x) / num_taps / num_branches)

    x_p = x[:W*num_taps*num_branches].reshape((W * num_taps, num_branches))
    h_p = pfb_window.reshape((num_taps, num_branches))

    output_rows = (W - 1) * num_taps
    x_summed = xp.zeros((output_rows, num_branches))
    for tap in range(num_taps):
        x_summed += x_p[tap:tap + output_rows, :] * h_p[tap, :]
    return x_summed


def get_pfb_window(num_taps: int,
                   num_branches: int,
                   window_fn: str='hamming') -> xp.ndarray:
    """Return PFB window coefficients.

    Args:
        num_taps: Number of PFB taps.
        num_branches: Number of PFB branches.
        window_fn: Windowing function used for the PFB.

    Returns:
        PFB window coefficients.
    """ 
    window = scipy.signal.firwin(num_taps * num_branches, 
                                 cutoff=1.0 / num_branches,
                                 window=window_fn,
                                 scale=True)
    window *= num_taps * num_branches
    return xp.array(window)


def get_pfb_voltages(x: xp.ndarray,
                     num_taps: int,
                     num_branches: int,
                     window_fn: str='hamming') -> xp.ndarray:
    """Produce coarse-channel complex voltages from real-voltage input.

    Args:
        x: Input voltage array.
        num_taps: Number of PFB taps.
        num_branches: Number of PFB branches.
        window_fn: Windowing function used for the PFB.

    Returns:
        Post-FFT complex voltages.
    """
    # Generate window coefficients
    win_coeffs = get_pfb_window(num_taps, num_branches, window_fn)
    
    # Apply frontend, take FFT, then take power (i.e. square)
    x_fir = pfb_frontend(x, win_coeffs, num_taps, num_branches)
    X_pfb = xp.fft.rfft(x_fir, num_branches, axis=1) / num_branches**0.5
    return X_pfb
