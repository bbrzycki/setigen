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


def render(fr, use_db=False, cb=True):
    """
    Display frame data in waterfall format.
    
    Parameters
    ----------
    fr : Frame, or 2D ndarray
        Input frame or Numpy array
    use_db : bool
        Option to convert intensities to dB.
    cb : bool
        Whether to display colorbar
    """ 
    # If `data` is a Frame object, just grab its data
    data = array(fr)
    if use_db:
        data = db(data)
    plt.imshow(data,
               aspect='auto',
               interpolation='none')
    if cb:
        plt.colorbar()
    plt.xlabel('Frequency (px)')
    plt.ylabel('Time (px)')
    
    
def plot(fr, use_db=False, cb=True):
    """
    Display frame data in waterfall format.
    
    Parameters
    ----------
    fr : Frame, or 2D ndarray
        Input frame or Numpy array
    use_db : bool
        Option to convert intensities to dB.
    cb : bool
        Whether to display colorbar
    """ 
    render(fr=fr, use_db=use_db, cb=cb)
    

def integrate(fr, axis='t', mode='mean', normalize=False):
    """
    Integrate along either time ('t', 0) or frequency ('f', 1) axes, to create 
    spectra or time series data. Mode is either 'mean' or 'sum'.
    
    Parameters
    ----------
    fr : Frame, or 2D ndarray
        Input frame or Numpy array
    axis : int or str
        Axis over which to integrate; time ('t', 0) or frequency ('f', 1)
    mode : str
        Integration mode, 'mean' or 'sum'
    normalize : bool
        Whether to normalize integrated array to mean 0, std 1
    
    Returns
    -------
    output : ndarray
        Integrated product
    """
    # If `data` is a Frame object, just grab its data
    data = array(fr)
    if axis in ['f', 1]:
        axis = 1
    else:
        axis = 0
        
    if mode[0] == 's':
        output = np.sum(data, axis=axis)
    else:
        output = np.mean(data, axis=axis)
        
    if normalize:
        c_output = sigma_clip(output)
        output = (output - np.mean(c_output)) / np.std(c_output)
        
    return output


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
