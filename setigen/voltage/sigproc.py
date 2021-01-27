import numpy as np
import scipy.signal


class PolyphaseFilterbank(object):
    """
    Creates polyphase filterbank for coarse channelization of real voltage input data.
    """
    
    def __init__(self, num_taps=8, num_branches=1024):
        self.num_taps = num_taps
        self.num_branches = num_branches
        
        self.cache = [None, None]
        
        self._get_pfb_window()
        
    def _get_pfb_window(self):
        self.window = get_pfb_window(self.num_taps, self.num_branches)
        
    def _pfb_frontend(self, x, pol=0):
        """
        Apply windowing function to create polyphase filterbank frontend.
        
        pol is either 0 or 1, for x and y polarizations.
        """
        # Cache last section of data, which is excluded in PFB step
        if self.cache[pol] is not None:
            x = np.concatenate([self.cache[pol], x])
        self.cache[pol] = x[-self.num_taps*self.num_branches:]
        
        return pfb_frontend(x, self.window, self.num_taps, self.num_branches)
        
    def channelize(self, x, pol=0):
        x_pfb = np.fft.rfft(self._pfb_frontend(x, pol=pol), 
                            self.num_branches,
                            axis=1) / self.num_branches**0.5
        return x_pfb
    

class RealQuantizer(object):
    def __init__(self, target_fwhm=32, num_bits=8):
        self.target_fwhm = target_fwhm
        self.num_bits = num_bits
        
    def quantize(self, voltages):
        return quantize_real(voltages,
                             target_fwhm=32,
                             num_bits=8)
    
    def digitize(self, voltages):
        return self.quantize(voltages)
    
    
class ComplexQuantizer(object):
    def __init__(self, target_fwhm=32, num_bits=8):
        self.target_fwhm = target_fwhm
        self.num_bits = num_bits
        
    def quantize(self, voltages):
        return quantize_complex(voltages,
                                target_fwhm=32,
                                num_bits=8)
    

def pfb_frontend(x, pfb_window, num_taps, num_branches):
    """
    Apply windowing function to create polyphase filterbank frontend.
    
    Follows description in Danny C. Price, Spectrometers and Polyphase 
    Filterbanks in Radio Astronomy, 2016. Available online at: 
    http://arxiv.org/abs/1607.03579.
    """
    W = int(len(x) / num_taps / num_branches)
    
    # Truncate data stream x to fit reshape step
    x_p = x[:W*num_taps*num_branches].reshape((W * num_taps, num_branches))
    h_p = pfb_window.reshape((num_taps, num_branches))
    
    # Resulting summed data array will be slightly shorter from windowing coeffs
    x_summed = np.zeros(((W - 1) * num_taps, num_branches))
    for t in range(0, (W - 1) * num_taps):
        x_weighted = x_p[t:t+num_taps, :] * h_p
        x_summed[t, :] = x_weighted.sum(axis=0)
    return x_summed

def get_pfb_window(num_taps, num_branches, window_fn='hamming'):
    """
    Get windowing function to multiply to time series data
    according to a finite impulse response (FIR) filter.
    """ 
    window = scipy.signal.get_window(window_fn, num_taps * num_branches)
    sinc = scipy.signal.firwin(num_taps * num_branches, 
                               cutoff=1.0 / num_branches,
                               window='rectangular')
    window *= sinc * num_branches
    return window

def get_pfb_voltages(x, num_taps, num_branches, window_fn='hamming'):
    """
    Produce complex raw voltage data as a function of time and coarse channel.
    """
    # Generate window coefficients
    win_coeffs = get_pfb_window(num_taps, num_branches, window_fn)
    
    # Apply frontend, take FFT, then take power (i.e. square)
    x_fir = pfb_frontend(x, win_coeffs, num_taps, num_branches)
    x_pfb = np.fft.rfft(x_fir, num_branches, axis=1) / num_branches**0.5
    return x_pfb

def get_pfb_waterfall(pfb_voltages, int_factor, fftlength):
    """
    Perform fine channelization on input complex voltage after filterbank,
    for a single polarization. 
    
    int_factor specifies the number of time samples to integrate.
    """
    
    X_samples = pfb_voltages.T
    X_samples = X_samples[:, :np.round(X_samples.shape[1] // fftlength) * fftlength]
    X_samples = X_samples.reshape((num_channels, X_samples.shape[1] // fftlength, fftlength))

    XX = np.fft.fft(X_samples, fftlength, axis=2) 
    XX = np.fft.fftshift(XX, axes=2)
    XX_psd = np.abs(XX)**2 / fftlength

    XX_psd = np.concatenate(XX_psd, axis=1)
    
    # Integrate over time, trimming if necessary
    XX_psd = XX_psd[:np.round(XX_psd.shape[0] // int_factor) * int_factor]
    XX_psd = XX_psd.reshape(XX_psd.shape[0] // int_factor, int_factor, XX_psd.shape[1])
    XX_psd = XX_psd.mean(axis=1)
    
    return XX_psd


def quantize_real(x, target_fwhm=32, num_bits=8):
    """
    Quantize real voltage data to integers with specified number of bits
    and target FWHM range. 
    """
#     # Estimate sigma quickly
#     data_sigma = np.std(pfb_voltages.flatten()[:10000])
    data_sigma = np.std(x.flatten())
    data_fwhm = 2 * np.sqrt(2 * np.log(2)) * data_sigma
    
    factor = target_fwhm / data_fwhm
    
    q_voltages = np.round(factor * x)
    q_voltages[q_voltages < -2**(num_bits - 1)] = -2**(num_bits - 1)
    q_voltages[q_voltages > 2**(num_bits - 1) - 1] = 2**(num_bits - 1) - 1
    q_voltages = q_voltages.astype(int)
    return q_voltages


def quantize_complex(x, target_fwhm=32, num_bits=8):
    """
    Quantize complex voltage data to integers with specified number of bits
    and target FWHM range. 
    """
    r, i = np.real(x), np.imag(x)
    q_r = quantize_real(r, target_fwhm=target_fwhm, num_bits=num_bits)
    q_i = quantize_real(i, target_fwhm=target_fwhm, num_bits=num_bits)
    return q_r + q_i * 1j