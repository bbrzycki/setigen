try:
    import cupy as xp
except ImportError:
    import numpy as xp
import numpy as np
import scipy.signal
import time



class PolyphaseFilterbank(object):
    """
    Creates polyphase filterbank for coarse channelization of real voltage input data.
    """
    
    def __init__(self, num_taps=8, num_branches=1024, window_fn='hamming'):
        self.num_taps = num_taps
        self.num_branches = num_branches
        self.window_fn = window_fn
        
        self.cache = [[None, None]]
        
        self._get_pfb_window()
        
    def _get_pfb_window(self):
        self.window = get_pfb_window(self.num_taps, self.num_branches, self.window_fn)
        
        # Somewhat arbitrary length to calculate spectral response, representing 
        # fftlength in fine channelization. Only needed to estimate peak to mean response
        length = 64 * self.num_taps
        freq_response_x = xp.zeros(self.num_branches * length)
        freq_response_x[:self.num_taps*self.num_branches] = self.window
        h = xp.fft.fft(freq_response_x)
        half_coarse_chan = (xp.abs(h)**2)[:length//2]+(xp.abs(h)**2)[length//2:length][::-1]
        self.max_mean_ratio = xp.max(half_coarse_chan) / xp.mean(half_coarse_chan)
        
        
    def _pfb_frontend(self, x, pol=0, antenna=0):
        """
        Apply windowing function to create polyphase filterbank frontend.
        
        pol is either 0 or 1, for x and y polarizations.
        """
        # Cache last section of data, which is excluded in PFB step
        if self.cache[antenna][pol] is not None:
            x = xp.concatenate([self.cache[antenna][pol], x])
        self.cache[antenna][pol] = x[-self.num_taps*self.num_branches:]
        
        return pfb_frontend(x, self.window, self.num_taps, self.num_branches)
        
    def channelize(self, x, pol=0, antenna=0):
#         start = time.time()
        x = self._pfb_frontend(x, pol=pol, antenna=antenna)
#         print('--pfb',time.time() - start)
#         start = time.time()
        x_pfb = xp.fft.rfft(x, 
                            self.num_branches,
                            axis=1) / self.num_branches**0.5
#         print('--rfft',time.time() - start)
#         start = time.time()
        return x_pfb
    

class RealQuantizer(object):
    def __init__(self, target_fwhm=32, num_bits=8):
        self.target_fwhm = target_fwhm
        self.target_sigma = self.target_fwhm / (2 * xp.sqrt(2 * xp.log(2)))
        self.num_bits = num_bits
        
#     def quantize(self, voltages):
#         return quantize_real(voltages,
#                              target_fwhm=self.target_fwhm,
#                              num_bits=self.num_bits)
    
    def quantize(self, voltages):
        x = voltages
        target_fwhm = self.target_fwhm
        num_bits = self.num_bits
        
        std_len = xp.amin(xp.array([10000, len(x)//10]))
        data_sigma = xp.std(x[:std_len])
        data_mean = xp.mean(x[:std_len])
        
        self.data_sigma = data_sigma
        self.data_mean = data_mean

        data_fwhm = 2 * xp.sqrt(2 * xp.log(2)) * data_sigma

        factor = target_fwhm / data_fwhm

        q_voltages = xp.around(factor * (x - data_mean))

        q_voltages[q_voltages < -2**(num_bits - 1)] = -2**(num_bits - 1)
        q_voltages[q_voltages > 2**(num_bits - 1) - 1] = 2**(num_bits - 1) - 1

    #     q_voltages = xp.where(q_voltages < -2**(num_bits - 1), -2**(num_bits - 1), q_voltages)
    #     q_voltages = xp.where(q_voltages > 2**(num_bits - 1) - 1, 2**(num_bits - 1) - 1, q_voltages)

        q_voltages = q_voltages.astype(int)

        return q_voltages
    
    def digitize(self, voltages):
        return self.quantize(voltages)
    
    
class ComplexQuantizer(object):
    def __init__(self, target_fwhm=32, num_bits=8):
        self.target_fwhm = target_fwhm
        self.target_sigma = self.target_fwhm / (2 * xp.sqrt(2 * xp.log(2)))
        self.num_bits = num_bits
        
    def quantize(self, voltages):
        return quantize_complex(voltages,
                                target_fwhm=self.target_fwhm,
                                num_bits=self.num_bits)
    
    
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
    """ 
    window = scipy.signal.firwin(num_taps * num_branches, 
                                 cutoff=1.0 / num_branches,
                                 window=window_fn,
                                 scale=True)
    window *= num_taps * num_branches
    return xp.asarray(window)


def get_pfb_voltages(x, num_taps, num_branches, window_fn='hamming'):
    """
    Produce complex raw voltage data as a function of time and coarse channel.
    """
    # Generate window coefficients
    win_coeffs = get_pfb_window(num_taps, num_branches, window_fn)
    
    # Apply frontend, take FFT, then take power (i.e. square)
    x_fir = pfb_frontend(x, win_coeffs, num_taps, num_branches)
    x_pfb = xp.fft.rfft(x_fir, num_branches, axis=1) / num_branches**0.5
    return x_pfb


def get_pfb_waterfall(pfb_voltages_x, pfb_voltages_y=None, int_factor=1, fftlength=256):
    """
    Perform fine channelization on input complex voltages after filterbank,
    for single or dual polarizations. 
    
    int_factor specifies the number of time samples to integrate.
    
    Shape of pfb_voltages is (time_samples, num_channels).
    """
    
    XX_psd = xp.zeros((pfb_voltages_x.shape[1], pfb_voltages_x.shape[0] // fftlength, fftlength))
    
    pfb_voltages_list = [pfb_voltages_x]
    if pfb_voltages_y is not None:
        pfb_voltages_list.append(pfb_voltages_y)
        
    for pfb_voltages in pfb_voltages_list:
        X_samples = pfb_voltages.T
        X_samples = X_samples[:, :(X_samples.shape[1] // fftlength) * fftlength]
        X_samples = X_samples.reshape((X_samples.shape[0], X_samples.shape[1] // fftlength, fftlength))
        XX = xp.fft.fft(X_samples, fftlength, axis=2) / fftlength**0.5
        XX = xp.fft.fftshift(XX, axes=2)
        XX_psd += xp.abs(XX)**2 

    XX_psd = xp.concatenate(XX_psd, axis=1)
    
    # Integrate over time, trimming if necessary
    XX_psd = XX_psd[:(XX_psd.shape[0] // int_factor) * int_factor]
    XX_psd = XX_psd.reshape(XX_psd.shape[0] // int_factor, int_factor, XX_psd.shape[1])
    XX_psd = XX_psd.sum(axis=1)
    
    return XX_psd


def get_waterfall_from_raw(raw_filename, block_size, num_chans, int_factor=1, fftlength=256):
    # produces waterfall from a raw file (only the first block), 2pol, 8bit
    with open(raw_filename, "rb") as f:
        i = 1
        chunk = f.read(80)
        while f"{'END':<80}".encode() not in chunk:
            chunk = f.read(80)
            i += 1
        # Skip zero padding
        chunk = f.read((512 - (80 * i % 512)))
        # Read data
        chunk = f.read(block_size)
        
    rawbuffer = np.frombuffer(chunk, dtype=xp.int8).reshape((num_chans, -1))
    rawbuffer_x = rawbuffer[:, 0::4] + rawbuffer[:, 1::4] * 1j
    rawbuffer_y = rawbuffer[:, 2::4] + rawbuffer[:, 3::4] * 1j    
    return get_pfb_waterfall(rawbuffer_x.T, rawbuffer_y.T, int_factor, fftlength)


def quantize_real(x, target_fwhm=32, num_bits=8):
    """
    Quantize real voltage data to integers with specified number of bits
    and target FWHM range. 
    """
#     # Estimate sigma quickly
#     data_sigma = xp.std(pfb_voltages.flatten()[:10000])
    start = time.time()
    
    std_len = xp.amin(xp.array([10000, len(x)//10]))
    data_sigma = xp.std(x[:std_len])
    data_mean = xp.mean(x[:std_len])
    
    data_fwhm = 2 * xp.sqrt(2 * xp.log(2)) * data_sigma
    
    factor = target_fwhm / data_fwhm
    
    q_voltages = xp.around(factor * (x - data_mean))
        
    q_voltages[q_voltages < -2**(num_bits - 1)] = -2**(num_bits - 1)
    q_voltages[q_voltages > 2**(num_bits - 1) - 1] = 2**(num_bits - 1) - 1
    
#     q_voltages = xp.where(q_voltages < -2**(num_bits - 1), -2**(num_bits - 1), q_voltages)
#     q_voltages = xp.where(q_voltages > 2**(num_bits - 1) - 1, 2**(num_bits - 1) - 1, q_voltages)
    
    q_voltages = q_voltages.astype(int)
    
    return q_voltages


def quantize_complex(x, target_fwhm=32, num_bits=8):
    """
    Quantize complex voltage data to integers with specified number of bits
    and target FWHM range. 
    """
    r, i = xp.real(x), xp.imag(x)
    q_r = quantize_real(r, target_fwhm=target_fwhm, num_bits=num_bits)
    q_i = quantize_real(i, target_fwhm=target_fwhm, num_bits=num_bits)
    return q_r + q_i * 1j