import numpy as np
import scipy.signal


def pfb_frontend(x, pfb_window, n_taps, n_chan):
    """
    Apply windowing function to create polyphase filterbank frontend.
    
    Follows description in Danny C. Price, Spectrometers and Polyphase Filterbanks in Radio Astronomy, 2016. Available online at: http://arxiv.org/abs/1607.03579.
    """
    W = int(len(x) / n_taps / n_chan)
    
    # Truncate data stream x to fit reshape step
    x_p = x[:W * n_taps * n_chan].reshape((W * n_taps, n_chan))
    h_p = pfb_window.reshape((n_taps, n_chan))
    
    # Resulting summed data array will be slightly shorter from windowing coeffs
    x_summed = np.zeros(((W - 1) * n_taps + 1, n_chan))
    for t in range(0, (W - 1) * n_taps + 1):
        x_weighted = x_p[t:t + n_taps, :] * h_p
        x_summed[t, :] = x_weighted.sum(axis=0)
    return x_summed

def get_pfb_window(n_taps, n_chan, window_fn='hamming'):
    """
    Get windowing function to multiply to time series data
    according to a finite impulse response (FIR) filter.
    """ 
    window = scipy.signal.get_window(window_fn, n_taps * n_chan)
    sinc = scipy.signal.firwin(n_taps * n_chan, 
                               cutoff=1.0 / n_chan,
                               window='rectangular')
    window *= sinc * n_chan
    return window

def get_pfb_voltages(x, n_taps, n_chan, window_fn='hamming'):
    """
    Produce complex raw voltage data as a function of time and coarse channel.
    """
    # Generate window coefficients
    win_coeffs = get_pfb_window(n_taps, n_chan, window_fn)
#     win_coeffs /= np.max(win_coeffs)
    # Apply frontend, take FFT, then take power (i.e. square)
    x_fir = pfb_frontend(x, win_coeffs, n_taps, n_chan)
    x_pfb = np.fft.rfft(x_fir, n_chan, axis=1) / n_chan**0.5
    return x_pfb

def get_pfb_waterfall(pfb_voltages, n_int, fftlength, start_channel, num_channels):
#     pfb_voltages[:, 204] += 10

    X_samples = pfb_voltages[:, start_channel:(start_channel + num_channels)]
    X_samples = X_samples.T
    X_samples = X_samples[:, :np.round(X_samples.shape[1] // fftlength) * fftlength]
    X_samples = X_samples.reshape((num_channels, X_samples.shape[1] // fftlength, fftlength))

    XX = np.fft.fft(X_samples, fftlength, axis=2) 
    XX = np.fft.fftshift(XX, axes=2)
    XX_psd = np.abs(XX)**2 / fftlength

    XX_psd = np.concatenate(XX_psd, axis=1)
    
    # Integrate over time, trimming if necessary
    XX_psd = XX_psd[:np.round(XX_psd.shape[0] // n_int) * n_int]
    XX_psd = XX_psd.reshape(XX_psd.shape[0] // n_int, n_int, XX_psd.shape[1])
    XX_psd = XX_psd.mean(axis=1)
    
    return XX_psd


def quantize_real(x, target_fwhm=30, n_bits=8):
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
    q_voltages[q_voltages < -2**(n_bits - 1)] = -2**(n_bits - 1)
    q_voltages[q_voltages > 2**(n_bits - 1) - 1] = 2**(n_bits - 1) - 1
    q_voltages = q_voltages.astype(int)
    return q_voltages


def quantize_complex(x, target_fwhm=30, n_bits=8):
    """
    Quantize complex voltage data to integers with specified number of bits
    and target FWHM range. 
    """
    r, i = np.real(x), np.imag(x)
    q_r = quantize_real(r, target_fwhm=target_fwhm, n_bits=n_bits)
    q_i = quantize_real(i, target_fwhm=target_fwhm, n_bits=n_bits)
    return q_r + q_i * 1j