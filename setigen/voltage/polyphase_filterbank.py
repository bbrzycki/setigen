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


class PolyphaseFilterbank(object):
    """
    Implement a polyphase filterbank (PFB) for coarse channelization of real voltage input data.
    
    Follows description in Danny C. Price, Spectrometers and Polyphase Filterbanks in 
    Radio Astronomy, 2016. Available online at: http://arxiv.org/abs/1607.03579.
    """
    
    def __init__(self, num_taps=8, num_branches=1024, window_fn='hamming'):
        """
        Initialize a polyphase filterbank object, with a voltage sample cache that ensures that
        consecutive sample retrievals get contiguous data (i.e. without introduced time delays).

        Parameters
        ----------
        num_taps : int, optional
            Number of PFB taps
        num_branches : int, optional
            Number of PFB branches. Note that this results in `num_branches / 2` coarse channels.
        window_fn : str, optional
            Windowing function used for the PFB
        """
        self.num_taps = num_taps
        self.num_branches = num_branches
        self.window_fn = window_fn
        
        self.cache = None
        
        self._get_pfb_window()
        
        # Estimate stds after channelizing Gaussian with mean 0, std 1
        self._get_channelized_stds()
        
    def _reset_cache(self):
        """
        Clear sample cache.
        """
        self.cache = None
                
    def _get_channelized_stds(self):
        """
        Estimate standard deviations in real and imaginary components after channelizing
        a zero-mean Gaussian distribution with variance 1. 
        """
        sample_v = xp.random.normal(0, 1, self.num_branches * 10000)
        v_pfb = self.channelize(sample_v, use_cache=False)
        self.channelized_stds = xp.array([v_pfb.real.std(), v_pfb.imag.std()])
        
    def _get_pfb_window(self):
        """
        Creates and saves PFB windowing coefficients. Saves frequency response shape
        and ratio of maximum to mean of the frequency response.
        """
        self.window = get_pfb_window(self.num_taps, self.num_branches, self.window_fn)
        
        # Somewhat arbitrary length to calculate spectral response, representing 
        # fftlength in fine channelization. Only needed to estimate peak to mean response
        length = 64 * self.num_taps
        freq_response_x = xp.zeros(self.num_branches * length)
        freq_response_x[:self.num_taps*self.num_branches] = self.window
        h = xp.fft.fft(freq_response_x)
        half_coarse_chan = (xp.abs(h)**2)[:length//2]+(xp.abs(h)**2)[length//2:length][::-1]
        self.response = self.half_coarse_chan = half_coarse_chan
        self.max_mean_ratio = xp.max(half_coarse_chan) / xp.mean(half_coarse_chan)
        
    def channelize(self, x, use_cache=True):
        """
        Channelize input voltages by applying the PFB and taking a normalized FFT. 

        Parameters
        ----------
        x : array
            Array of voltages
            
        Returns
        -------
        X_pfb : array
            Post-FFT complex voltages
        """
        if use_cache:
            # Cache last section of data, which is excluded in PFB step
            if self.cache is not None:
                x = xp.concatenate([self.cache, x])
            self.cache = x[-self.num_taps*self.num_branches:]
        
        x = pfb_frontend(x, self.window, self.num_taps, self.num_branches)
        X_pfb = xp.fft.fft(x, 
                           self.num_branches,
                           axis=1)[:, 0:self.num_branches//2] / self.num_branches**0.5
        return X_pfb
    
    
def pfb_frontend(x, pfb_window, num_taps, num_branches):
    """
    Apply windowing function to create polyphase filterbank frontend.
    
    Follows description in Danny C. Price, Spectrometers and Polyphase 
    Filterbanks in Radio Astronomy, 2016. Available online at: 
    http://arxiv.org/abs/1607.03579.
    
    Parameters
    ----------
    x : array
        Array of voltages
    pfb_window : array
        Array of PFB windowing coefficients
    num_taps : int
        Number of PFB taps
    num_branches : int
        Number of PFB branches. Note that this results in `num_branches / 2` coarse channels.
            
    Returns
    -------
    x_summed : array
        Array of voltages post-PFB weighting
    """
    W = int(len(x) / num_taps / num_branches)
    
    # Truncate data stream x to fit reshape step
    x_p = x[:W*num_taps*num_branches].reshape((W * num_taps, num_branches))
    h_p = pfb_window.reshape((num_taps, num_branches))
    
    # Resulting summed data array will be slightly shorter from windowing coeffs
    I = xp.expand_dims(xp.arange(num_taps), 0) + xp.expand_dims(xp.arange((W - 1) * num_taps), 0).T
    x_summed = xp.sum(x_p[I] * h_p, axis=1) / num_taps
    
#     x_summed = xp.zeros(((W - 1) * num_taps, num_branches))
#     for t in range(0, (W - 1) * num_taps):
#         x_weighted = x_p[t:t+num_taps, :] * h_p
#         x_summed[t, :] = xp.sum(x_weighted, axis=0)
    return x_summed


def get_pfb_window(num_taps, num_branches, window_fn='hamming'):
    """
    Get windowing function to multiply to time series data
    according to a finite impulse response (FIR) filter.

    Parameters
    ----------
    num_taps : int
        Number of PFB taps
    num_branches : int
        Number of PFB branches. Note that this results in `num_branches / 2` coarse channels.
    window_fn : str, optional
        Windowing function used for the PFB
            
    Returns
    -------
    window : array
        Array of PFB windowing coefficients
    """ 
    window = scipy.signal.firwin(num_taps * num_branches, 
                                 cutoff=1.0 / num_branches,
                                 window=window_fn,
                                 scale=True)
    window *= num_taps * num_branches
    return xp.array(window)


def get_pfb_voltages(x, num_taps, num_branches, window_fn='hamming'):
    """
    Produce complex raw voltage data as a function of time and coarse channel.

    Parameters
    ----------
    x : array
        Array of voltages
    num_taps : int
        Number of PFB taps
    num_branches : int
        Number of PFB branches. Note that this results in `num_branches / 2` coarse channels.
    window_fn : str, optional
        Windowing function used for the PFB
            
    Returns
    -------
    X_pfb : array
        Post-FFT complex voltages
    """
    # Generate window coefficients
    win_coeffs = get_pfb_window(num_taps, num_branches, window_fn)
    
    # Apply frontend, take FFT, then take power (i.e. square)
    x_fir = pfb_frontend(x, win_coeffs, num_taps, num_branches)
    X_pfb = xp.fft.rfft(x_fir, num_branches, axis=1) / num_branches**0.5
    return X_pfb
