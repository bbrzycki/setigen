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
    def __init__(self,
                 target_fwhm=32, 
                 num_bits=8, 
                 stats_calc_freq=1,
                 stats_calc_num_samples=10000):
        self.target_fwhm = target_fwhm
        self.target_sigma = self.target_fwhm / (2 * xp.sqrt(2 * xp.log(2)))
        self.num_bits = num_bits
        
        self.data_mean = None
        self.data_sigma = None
        
        self.stats_calc_index = 0
        self.stats_calc_freq = stats_calc_freq
        self.stats_calc_num_samples = stats_calc_num_samples
    
    def quantize(self, voltages):
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
        if self.stats_calc_index == self.stats_calc_freq:
            self.stats_calc_index = 0

        return q_voltages
    
    def digitize(self, voltages):
        return self.quantize(voltages)
    
    
class ComplexQuantizer(object):
    def __init__(self,
                 target_fwhm=32, 
                 num_bits=8,
                 stats_calc_freq=1,
                 stats_calc_num_samples=10000):
        self.target_fwhm = target_fwhm
        self.target_sigma = self.target_fwhm / (2 * xp.sqrt(2 * xp.log(2)))
        self.num_bits = num_bits
        
        self.data_mean_r = None
        self.data_sigma_r = None
        self.data_mean_i = None
        self.data_sigma_i = None
        
        self.stats_calc_freq = stats_calc_freq
        self.stats_calc_num_samples = stats_calc_num_samples
        
        self.quantizer_r = RealQuantizer(target_fwhm=target_fwhm,
                                         num_bits=num_bits,
                                         stats_calc_freq=stats_calc_freq,
                                         stats_calc_num_samples=stats_calc_num_samples)
        self.quantizer_i = RealQuantizer(target_fwhm=target_fwhm,
                                         num_bits=num_bits,
                                         stats_calc_freq=stats_calc_freq,
                                         stats_calc_num_samples=stats_calc_num_samples)
        
    def quantize(self, voltages):
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