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
        self.target_fwhm = target_fwhm
        self.target_sigma = self.target_fwhm / (2 * xp.sqrt(2 * xp.log(2)))
        self.num_bits = num_bits
        
        self.data_mean = None
        self.data_sigma = None
        
        self.stats_calc_index = 0
        self.stats_calc_period = stats_calc_period
        self.stats_calc_num_samples = stats_calc_num_samples
    
    def quantize(self, voltages):
        """
        Quantize input voltages. Save voltage mean and standard deviation as `self.data_mean` and
        `self.data_sigma` for convenience.
        
        Parameters
        ----------
        voltages : array
            Array of real voltages
            
        Returns
        -------
        q_voltages : array
            Array of quantized voltages
        """
        if self.stats_calc_index == 0:
            self.data_mean, self.data_sigma = data_stream.estimate_stats(voltages, 
                                                                         self.stats_calc_num_samples)
            
        q_voltages = quantize_real(voltages, 
                                   target_fwhm=self.target_fwhm,
                                   num_bits=self.num_bits,
                                   data_mean=self.data_mean,
                                   data_sigma=self.data_sigma,
                                   stats_calc_num_samples=self.stats_calc_num_samples)
        
        self.stats_calc_index += 1
        if self.stats_calc_index == self.stats_calc_period:
            self.stats_calc_index = 0

        return q_voltages
    
    def digitize(self, voltages):
        """
        Quantize input voltages. Wrapper for :code:`quantize()`.
        """
        return self.quantize(voltages)
    
    
class ComplexQuantizer(object):
    """
    Implement a quantizer for complex voltages, using a pair of RealQuantizers.
    """
    def __init__(self,
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
        self.target_fwhm = target_fwhm
        self.target_sigma = self.target_fwhm / (2 * xp.sqrt(2 * xp.log(2)))
        self.num_bits = num_bits
        
        self.data_mean_r = None
        self.data_sigma_r = None
        self.data_mean_i = None
        self.data_sigma_i = None
        
        self.stats_calc_period = stats_calc_period
        self.stats_calc_num_samples = stats_calc_num_samples
        
        self.quantizer_r = RealQuantizer(target_fwhm=target_fwhm,
                                         num_bits=num_bits,
                                         stats_calc_period=stats_calc_period,
                                         stats_calc_num_samples=stats_calc_num_samples)
        self.quantizer_i = RealQuantizer(target_fwhm=target_fwhm,
                                         num_bits=num_bits,
                                         stats_calc_period=stats_calc_period,
                                         stats_calc_num_samples=stats_calc_num_samples)
        
    def quantize(self, voltages):
        """
        Quantize input complex voltages. Save voltage mean and standard deviation for each component
        as `self.data_mean_r, self.data_mean_i` and `self.data_sigma_r, self.data_sigma_i` for convenience.
        
        Parameters
        ----------
        voltages : array
            Array of complex voltages
            
        Returns
        -------
        q_voltages : array
            Array of complex quantized voltages
        """
        q_r = self.quantizer_r.quantize(xp.real(voltages))
        q_i = self.quantizer_i.quantize(xp.imag(voltages))
        
        self.data_mean_r, self.data_sigma_r = self.quantizer_r.data_mean, self.quantizer_r.data_sigma
        self.data_mean_i, self.data_sigma_i = self.quantizer_i.data_mean, self.quantizer_i.data_sigma
        
        return q_r + q_i * 1j


def quantize_real(x,
                  target_fwhm=32, 
                  num_bits=8,
                  data_mean=None,
                  data_sigma=None,
                  stats_calc_num_samples=10000):
    """
    Quantize real voltage data to integers with specified number of bits
    and target FWHM range. 

    Parameters
    ----------
    x : array
        Array of voltages
    target_fwhm : float, optional
        Target FWHM
    num_bits : int, optional
        Number of bits to quantize to. Quantized voltages will span -2**(num_bits - 1) 
        to 2**(num_bits - 1) - 1, inclusive.
    data_mean : float, optional
        Mean of input voltages, if already known
    data_sigma : float, optional
        Standard deviation of input voltages, if already known. If None, estimates mean and
        standard deviation automatically.
    stats_calc_num_samples : int, optional
        Maximum number of samples for use in estimating noise statistics
        
    Returns
    -------
    q_voltages : array
        Array of quantized voltages
    """
    if data_sigma is None:
        data_mean, data_sigma = data_stream.estimate_stats(x, stats_calc_num_samples)
    
    data_fwhm = 2 * xp.sqrt(2 * xp.log(2)) * data_sigma
    
    factor = target_fwhm / data_fwhm
    
    q_voltages = xp.around(factor * (x - data_mean))
    q_voltages = xp.clip(q_voltages, -2**(num_bits - 1), 2**(num_bits - 1) - 1)
    q_voltages = q_voltages.astype(int)
    
    return q_voltages


def quantize_complex(x, 
                     target_fwhm=32, 
                     num_bits=8, 
                     stats_calc_num_samples=10000):
    """
    Quantize complex voltage data to integers with specified number of bits
    and target FWHM range. 

    Parameters
    ----------
    x : array
        Array of complex voltages
    target_fwhm : float, optional
        Target FWHM
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
                        target_fwhm=target_fwhm, 
                        num_bits=num_bits,
                        stats_calc_num_samples=stats_calc_num_samples)
    
    q_i = quantize_real(i, 
                        target_fwhm=target_fwhm, 
                        num_bits=num_bits,
                        stats_calc_num_samples=stats_calc_num_samples)
    
    q_c = q_r + q_i * 1j
    
    return q_c