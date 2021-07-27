import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import sigma_clip


def db(a):
    """
    Convert to dB.
    """
    return 10 * np.log10(a)


def array(fr):
    """
    Return a Numpy array for input frame or Numpy array.
    
    Parameters
    ----------
    fr : Frame, or 2D ndarray
        Input frame or Numpy array
    
    Returns
    -------
    data : ndarray
        Data array
    """
    try:
        return fr.get_data()
    except AttributeError:
        return fr


def render(data, use_db=False, cb=True):
    """
    Display frame data in waterfall format.
    
    Parameters
    ----------
    data : Frame, or 2D ndarray
        Input frame or Numpy array
    use_db : bool
        Option to convert intensities to dB.
    cb : bool
        Whether to display colorbar
    """ 
    # If `data` is a Frame object, just grab its data
    data = array(data)
    if use_db:
        data = db(data)
    plt.imshow(data,
               aspect='auto',
               interpolation='none')
    if cb:
        plt.colorbar()
    plt.xlabel('Frequency (px)')
    plt.ylabel('Time (px)')
    
    
def plot(data, use_db=False, cb=True):
    """
    Display frame data in waterfall format.
    
    Parameters
    ----------
    data : Frame, or 2D ndarray
        Input frame or Numpy array
    use_db : bool
        Option to convert intensities to dB.
    cb : bool
        Whether to display colorbar
    """ 
    render(data=data, use_db=use_db, cb=cb)
    

def integrate(data, axis='t', mode='mean'):
    """
    Integrate along either time ('t', 0) or frequency ('f', 1) axes, to create 
    spectra or time series data. Mode is either 'mean' or 'sum'.
    
    Parameters
    ----------
    data : Frame, or 2D ndarray
        Input frame or Numpy array
    axis : int or str
        Axis over which to integrate; time ('t', 0) or frequency ('f', 1)
    mode : str
        Integration mode, 'mean' or 'sum'
    
    Returns
    -------
    output : ndarray
        Integrated product
    """
    # If `data` is a Frame object, just grab its data
    data = array(data)
    if axis in ['f', 1]:
        axis = 1
    else:
        axis = 0
        
    if mode[0] == 's':
        return np.sum(data, axis=axis)
    else:
        return np.mean(data, axis=axis)
    

def integrate_frame_subdata(data, frame=None, normalize=False):
    """
    Integrate a chunk of data assuming frame statistics using mean (not sum).
    """
    if normalize:
        assert frame is not None
        m, s = frame.get_noise_stats()
        data = (data - m) / (s / frame.tchans**0.5)
    return np.mean(data, axis=0)


def get_slice(fr, l, r):
    """
    Slice frame data with left and right index bounds.
    
    Parameters
    ----------
    fr : Frame
        Input frame
    l : int
        Left bound
    r : int
        Right bound
        
    Returns
    -------
    s_fr : Frame
        Sliced frame
    """
    s_data = fr.data[:, l:r]

    # Match frequency to truncated frame
    if fr.ascending:
        fch1 = fr.fs[l]
    else:
        fch1 = fr.fs[r - 1]

    s_fr = fr.from_data(fr.df, 
                        fr.dt, 
                        fch1, 
                        fr.ascending,
                        s_data,
                        metadata=fr.metadata,
                        waterfall=fr.check_waterfall())

    return s_fr
