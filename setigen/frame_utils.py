import numpy as np
import matplotlib.pyplot as plt


def db(x):
    """
    Converts to dB.
    """
    return 10 * np.log10(x)


def render(data, cb=True):
    """
    Display frame data in waterfall format.
    
    Parameters
    ----------
    data : 2D numpy array
    cb : bool
        Whether to display colorbar
    """ 
    plt.imshow(data,
               aspect='auto',
               interpolation='none')
    if cb:
        plt.colorbar()
    plt.xlabel('Frequency (px)')
    plt.ylabel('Time (px)')
    

def integrate_frame(frame, normalize=False):
    """
    Integrate over time using mean (not sum).
    """
    data = frame.data
    if normalize:
        m, s = frame.get_noise_stats()
        data = (data - m) / (s / frame.tchans**0.5)
    return np.mean(data, axis=0)


def integrate_frame_subdata(data, frame=None, normalize=False):
    """
    Integrate a chunk of data assuming frame statistics using mean (not sum).
    """
    if normalize:
        assert frame is not None
        m, s = frame.get_noise_stats()
        data = (data - m) / (s / frame.tchans**0.5)
    return np.mean(data, axis=0)