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
    
import numpy as np

from . import data_stream


class RealQuantizer(object):
    """Quantize real-valued voltages to integer levels."""
    
    def __init__(self,
                 target_mean: float=0,
                 target_fwhm: float=32, 
                 num_bits: int=8, 
                 stats_calc_period: int=1,
                 stats_calc_num_samples: int=10000) -> None:
        """Initialize a real-valued quantizer.

        Args:
            target_mean: Target mean of the quantized values.
            target_fwhm: Target full width at half maximum.
            num_bits: Number of quantization bits.
            stats_calc_period: Period for recomputing input statistics.
            stats_calc_num_samples: Maximum samples used for statistic estimates.
        """
        self.target_mean = target_mean
        self.target_fwhm = target_fwhm
        self.target_std = self.target_fwhm / (2 * np.sqrt(2 * np.log(2)))
        self.num_bits = num_bits
        
        self.stats_cache = [None, None]
        self.stats_calc_indices = 0 
        
        self.stats_calc_period = stats_calc_period
        self.stats_calc_num_samples = stats_calc_num_samples
        
    
    def _reset_cache(self) -> None:
        """Clear cached statistics and update counters."""
        self.stats_calc_indices = 0
        self.stats_cache = [None, None]
             
       
    def _set_target_stats(self, target_mean: float, target_std: float) -> None:
        """Set target mean and standard deviation for the quantizer.

        Args:
            target_mean: Target mean.
            target_std: Target standard deviation.
        """
        self.target_mean = target_mean
        self.target_std = target_std
        self.target_fwhm = target_std * (2 * np.sqrt(2 * np.log(2)))
        
    
    def quantize(self, voltages: xp.ndarray, custom_std: float | None = None) -> xp.ndarray:
        """Quantize real input voltages.

        Args:
            voltages: Array of real voltages.
            custom_std: Optional standard deviation to use for scaling.

        Returns:
            Array of quantized voltages.
        """
        if self.stats_calc_indices == 0:
            self.stats_cache = data_stream.estimate_stats(voltages, self.stats_calc_num_samples)
            
        if custom_std is not None:
            data_std = custom_std
        else:
            data_std = self.stats_cache[1]
        q_voltages = quantize_real(voltages, 
                                   target_mean=self.target_mean,
                                   target_std=self.target_std,
                                   num_bits=self.num_bits,
                                   data_mean=self.stats_cache[0],
                                   data_std=data_std,
                                   stats_calc_num_samples=self.stats_calc_num_samples)
        
        self.stats_calc_indices += 1
        if self.stats_calc_indices == self.stats_calc_period:
            self.stats_calc_indices = 0

        return q_voltages
    
    
    def digitize(self, voltages: xp.ndarray, custom_std: float | None = None) -> xp.ndarray:
        """Alias for `quantize()`.

        Args:
            voltages: Array of real voltages.
            custom_std: Optional standard deviation to use for scaling.

        Returns:
            Array of quantized voltages.
        """
        return self.quantize(voltages, custom_std=custom_std)
    
    
class ComplexQuantizer(object):
    """Quantize complex voltages using paired real-valued quantizers."""
    
    def __init__(self,
                 target_mean: float=0,
                 target_fwhm: float=32, 
                 num_bits: int=8,
                 stats_calc_period: int=1,
                 stats_calc_num_samples: int=10000) -> None:
        """Initialize a complex-valued quantizer.

        Args:
            target_mean: Target mean of the quantized values.
            target_fwhm: Target full width at half maximum.
            num_bits: Number of quantization bits.
            stats_calc_period: Period for recomputing input statistics.
            stats_calc_num_samples: Maximum samples used for statistic estimates.
        """
        self.target_mean = target_mean
        self.target_fwhm = target_fwhm
        self.target_std = self.target_fwhm / (2 * np.sqrt(2 * np.log(2)))
        self.num_bits = num_bits
        
        self.stats_cache_r = [None, None]
        self.stats_cache_i = [None, None] 
        
        self.stats_calc_period = stats_calc_period
        self.stats_calc_num_samples = stats_calc_num_samples
        
        self.quantizer_r = RealQuantizer(target_mean=target_mean,
                                         target_fwhm=target_fwhm,
                                         num_bits=num_bits,
                                         stats_calc_period=stats_calc_period,
                                         stats_calc_num_samples=stats_calc_num_samples)
        self.quantizer_i = RealQuantizer(target_mean=target_mean,
                                         target_fwhm=target_fwhm,
                                         num_bits=num_bits,
                                         stats_calc_period=stats_calc_period,
                                         stats_calc_num_samples=stats_calc_num_samples)
        
    
    def _reset_cache(self) -> None:
        """Clear cached statistics and update counters."""
        self.stats_cache_r = [None, None]
        self.stats_cache_i = [None, None]
        self.quantizer_r._reset_cache()
        self.quantizer_i._reset_cache()
        
    
    def quantize(self,
                 voltages: xp.ndarray,
                 custom_stds: float | list[float | None] | tuple[float | None, float | None] | None = None) -> xp.ndarray:
        """Quantize complex input voltages.

        Args:
            voltages: Array of complex voltages.
            custom_stds: Optional scalar or length-two sequence of standard
                deviations for the real and imaginary parts.

        Returns:
            Array of complex quantized voltages.

        Raises:
            ValueError: If `custom_stds` is not scalar or length two.
        """
        if custom_stds is None or np.isscalar(custom_stds):
            custom_stds = [custom_stds] * 2
        else:
            if len(custom_stds) != 2:
                raise ValueError("custom_stds must be a scalar or a length-2 sequence.")
            
        q_r = self.quantizer_r.quantize(xp.real(voltages), custom_std=custom_stds[0])
        q_i = self.quantizer_i.quantize(xp.imag(voltages), custom_std=custom_stds[1])
        
        self.stats_cache_r = self.quantizer_r.stats_cache
        self.stats_cache_i = self.quantizer_i.stats_cache
        
        return q_r + q_i * 1j


def quantize_real(x: xp.ndarray,
                  target_mean: float = 0,
                  target_std: float = 32 / (2 * np.sqrt(2 * np.log(2))),
                  num_bits: int = 8,
                  data_mean: float | None = None,
                  data_std: float | None = None,
                  stats_calc_num_samples: int = 10000) -> xp.ndarray:
    """Quantize real voltages to integer levels.

    Args:
        x: Array of voltages.
        target_mean: Target mean for quantized voltages.
        target_std: Target standard deviation for quantized voltages.
        num_bits: Number of quantization bits.
        data_mean: Optional precomputed input mean.
        data_std: Optional precomputed input standard deviation.
        stats_calc_num_samples: Maximum samples used for statistic estimates.

    Returns:
        Array of quantized voltages.
    """
    if data_std is None:
        data_mean, data_std = data_stream.estimate_stats(x, stats_calc_num_samples)
    
    if data_std == 0:
        factor = 0
    else:
        factor = target_std / data_std
    
    q_voltages = xp.around(factor * (x - data_mean) + target_mean)
    q_voltages = xp.clip(q_voltages, -2**(num_bits - 1), 2**(num_bits - 1) - 1)
    q_voltages = q_voltages.astype(int)
    
    return q_voltages


def quantize_complex(x: xp.ndarray,
                     target_mean: float = 0,
                     target_std: float = 32 / (2 * np.sqrt(2 * np.log(2))),
                     num_bits: int = 8,
                     stats_calc_num_samples: int = 10000) -> xp.ndarray:
    """Quantize complex voltages to integer levels.

    Args:
        x: Array of complex voltages.
        target_mean: Target mean for quantized voltages.
        target_std: Target standard deviation for quantized voltages.
        num_bits: Number of quantization bits.
        stats_calc_num_samples: Maximum samples used for statistic estimates.

    Returns:
        Array of complex quantized voltages.
    """
    r, i = xp.real(x), xp.imag(x)
    
    q_r = quantize_real(r, 
                        target_mean=target_mean,
                        target_std=target_std, 
                        num_bits=num_bits,
                        stats_calc_num_samples=stats_calc_num_samples)
    
    q_i = quantize_real(i, 
                        target_mean=target_mean,
                        target_std=target_std, 
                        num_bits=num_bits,
                        stats_calc_num_samples=stats_calc_num_samples)
    
    q_c = q_r + q_i * 1j
    
    return q_c
