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
import scipy.signal
import time

from . import data_stream


class RealQuantizer(object):
    """
    Implement a quantizer for input voltages.
    """
    def __init__(self,
                 target_mean=0,
                 target_fwhm=32, 
                 num_bits=8, 
                 stats_calc_period=1,
                 stats_calc_num_samples=10000):
        """
        Initialize a quantizer, which maps real input voltages to integers between
        -2**(num_bits - 1) and 2**(num_bits - 1) - 1, inclusive. Specifically, it estimates the
        mean and standard deviation of the voltages, and maps to 0 mean and a target full width at
        half maximum (FWHM). Voltages that extend past the quantized voltage range are clipped
        accordingly.
        
        The mean and standard deviation calculations can be limited to save computation using the 
        `stats_calc_period` and `stats_calc_num_samples` parameters. The former is an integer that
        specifies the period of computation; if 1, it computes the stats every time. If set to a 
        non-positive integer, like -1, the computation will run once during the first call and never
        again. The latter specifies the maximum number of voltage samples to use in calculating the 
        statistics; depending on the nature of the input voltages, a relatively small number of samples 
        may be sufficient for capturing the general distribution of voltages.

        Parameters
        ----------
        target_fwhm : float, optional
            Target FWHM
        num_bits : int, optional
            Number of bits to quantize to. Quantized voltages will span -2**(num_bits - 1) 
            to 2**(num_bits - 1) - 1, inclusive.
        stats_calc_period : int, optional
            Sets the period for computing the mean and standard deviation of input voltages
        stats_calc_num_samples : int, optional
            Maximum number of samples for use in estimating noise statistics
        """
        self.target_mean = target_mean
        self.target_fwhm = target_fwhm
        self.target_std = self.target_fwhm / (2 * xp.sqrt(2 * xp.log(2)))
        self.num_bits = num_bits
        
        self.stats_cache = [None, None]
        self.stats_calc_indices = 0 
        
        self.stats_calc_period = stats_calc_period
        self.stats_calc_num_samples = stats_calc_num_samples
        
    def _reset_cache(self):
        """
        Clear statistics and indices caches.
        """
        self.stats_calc_indices = 0
        self.stats_cache = [None, None]
                
    def _set_target_stats(self, target_mean, target_std):
        """
        Set the target stats for the quantizer.
        
        Parameters
        ----------
        target_mean : float
            Target mean
        target_std : float
            Target standard deviation
        """
        self.target_mean = target_mean
        self.target_std = target_std
        self.target_fwhm = target_std * (2 * xp.sqrt(2 * xp.log(2)))
        
    def quantize(self, voltages, custom_std=None):
        """
        Quantize input voltages. Cache voltage mean and standard deviation, per polarization and
        per antenna.
        
        Parameters
        ----------
        voltages : array
            Array of real voltages
        custom_std : float
            Custom standard deviation to use for scaling, instead of automatic calculation. The quantizer will go
            from custom_std to self.target_std. 
            
        Returns
        -------
        q_voltages : array
            Array of quantized voltages
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
    
    def digitize(self, voltages, custom_std=None):
        """
        Quantize input voltages. Wrapper for :code:`quantize()`.
        """
        return self.quantize(voltages, custom_std=custom_std)
    
    
class ComplexQuantizer(object):
    """
    Implement a quantizer for complex voltages, using a pair of RealQuantizers.
    """
    def __init__(self,
                 target_mean=0,
                 target_fwhm=32, 
                 num_bits=8,
                 stats_calc_period=1,
                 stats_calc_num_samples=10000):
        """
        Initialize a complex quantizer, which maps complex input voltage components to integers
        between -2**(num_bits - 1) and 2**(num_bits - 1) - 1, inclusive. Uses a pair of 
        RealQuantizers to quantize real and imaginary components separately. 

        Parameters
        ----------
        target_fwhm : float, optional
            Target FWHM
        num_bits : int, optional
            Number of bits to quantize to. Quantized voltages will span -2**(num_bits - 1) 
            to 2**(num_bits - 1) - 1, inclusive.
        stats_calc_period : int, optional
            Sets the period for computing the mean and standard deviation of input voltages
        stats_calc_num_samples : int, optional
            Maximum number of samples for use in estimating noise statistics
        """
        self.target_mean = target_mean
        self.target_fwhm = target_fwhm
        self.target_std = self.target_fwhm / (2 * xp.sqrt(2 * xp.log(2)))
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
        
    def _reset_cache(self):
        """
        Clear statistics and indices caches.
        """
        self.stats_cache_r = [None, None]
        self.stats_cache_i = [None, None]
        self.quantizer_r._reset_cache()
        self.quantizer_i._reset_cache()
        
    def quantize(self, voltages, custom_stds=None):
        """
        Quantize input complex voltages. Cache voltage means and standard deviations, per 
        polarization and per antenna.
        
        Parameters
        ----------
        voltages : array
            Array of complex voltages
        custom_stds : float, list, or array
            Custom standard deviation to use for scaling, instead of automatic calculation. Each quantizer will go
            from custom_stds values to self.target_std. Can either be a single value or an array-like object of length
            2, to set the custom standard deviation for real and imaginary parts.
            
        Returns
        -------
        q_voltages : array
            Array of complex quantized voltages
        """
        try:
            assert len(custom_stds) == 2
        except TypeError:
            custom_stds = [custom_stds] * 2
            
        q_r = self.quantizer_r.quantize(xp.real(voltages), custom_std=custom_stds[0])
        q_i = self.quantizer_i.quantize(xp.imag(voltages), custom_std=custom_stds[1])
        
        self.stats_cache_r = self.quantizer_r.stats_cache
        self.stats_cache_i = self.quantizer_i.stats_cache
        
        return q_r + q_i * 1j


def quantize_real(x,
                  target_mean=0,
                  target_std=32/(2*xp.sqrt(2*xp.log(2))), 
                  num_bits=8,
                  data_mean=None,
                  data_std=None,
                  stats_calc_num_samples=10000):
    """
    Quantize real voltage data to integers with specified number of bits
    and target statistics. 

    Parameters
    ----------
    x : array
        Array of voltages
    target_mean : float, optional
        Target mean for voltages
    target_std : float, optional
        Target standard deviation for voltages
    num_bits : int, optional
        Number of bits to quantize to. Quantized voltages will span -2**(num_bits - 1) 
        to 2**(num_bits - 1) - 1, inclusive.
    data_mean : float, optional
        Mean of input voltages, if already known
    data_std : float, optional
        Standard deviation of input voltages, if already known. If None, estimates mean and
        standard deviation automatically.
    stats_calc_num_samples : int, optional
        Maximum number of samples for use in estimating noise statistics
        
    Returns
    -------
    q_voltages : array
        Array of quantized voltages
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


def quantize_complex(x, 
                     target_mean=0,
                     target_std=32/(2*xp.sqrt(2*xp.log(2))), 
                     num_bits=8, 
                     stats_calc_num_samples=10000):
    """
    Quantize complex voltage data to integers with specified number of bits
    and target FWHM range. 

    Parameters
    ----------
    x : array
        Array of complex voltages
    target_mean : float, optional
        Target mean for voltages
    target_std : float, optional
        Target standard deviation for voltages
    num_bits : int, optional
        Number of bits to quantize to. Quantized voltages will span -2**(num_bits - 1) 
        to 2**(num_bits - 1) - 1, inclusive.
    stats_calc_num_samples : int, optional
        Maximum number of samples for use in estimating noise statistics
        
    Returns
    -------
    q_c : array
        Array of complex quantized voltages
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