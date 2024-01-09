import numpy as np
    

def _copy_docstring(copy_func):
    """
    Copy plotting docstring, for convenience in class plotting methods.
    """
    def wrapped(func):
        func.__doc__ = copy_func.__doc__ 
        return func
    return wrapped


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