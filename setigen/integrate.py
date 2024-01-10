import numpy as np
from astropy.stats import sigma_clip
from . import utils
from .spectrum import Spectrum 
from .timeseries import TimeSeries
    

def integrate(fr, axis='t', mode='mean', normalize=False, as_frame=False):
    """
    Integrate along either time ('t', 0) or frequency ('f', 1) axes, to create 
    spectra or time series data. Mode is either 'mean' or 'sum'.
    
    Parameters
    ----------
    fr : Frame, or 2D ndarray
        Input frame or Numpy array
    axis : int or str
        Axis over which to integrate; time ('t', 0) or frequency ('f', 1)
    mode : {"mean", "sum"}, default: "mean"
        Integration mode
    normalize : bool
        Option to normalize integrated array to mean 0, std 1
    as_frame : bool
        Option to format result as a frame
    
    Returns
    -------
    output : ndarray or Frame
        Integrated intensities
    """
    # If `data` is a Frame object, just grab its data
    data = utils.array(fr)
    if axis in ['f', 1]:
        # Time series
        axis = 1
    else:
        # Spectrum
        axis = 0
        
    if mode[0] == 's':
        data = np.sum(data, axis=axis, keepdims=True)
    else:
        data = np.mean(data, axis=axis, keepdims=True)
        
    if normalize:
        c_data = sigma_clip(data)
        data = (data - np.mean(c_data)) / np.std(c_data)

    if as_frame:
        if axis in ['f', 1]:
            # Time series
            new_fr = TimeSeries(df=fr.df * fr.fchans,
                                dt=fr.dt,
                                fch1=fr.fmid,
                                ascending=fr.ascending,
                                data=data,
                                seed=fr.rng)
        else:
            # Spectrum
            new_fr = Spectrum(df=fr.df,
                              dt=fr.dt * fr.tchans,
                              fch1=fr.fch1,
                              ascending=fr.ascending,
                              data=data,
                              seed=fr.rng)
        return new_fr
    else:
        return data.flatten()


def spectrum(fr, mode="mean", normalize=False):
    """
    Produce default Spectrum object from spectrogram Frame.
    """
    return integrate(fr, axis=0, mode=mode, normalize=normalize, as_frame=True) 


def timeseries(fr, mode="mean", normalize=False):
    """
    Produce default TimeSeries object from spectrogram Frame.
    """
    return integrate(fr, axis=1, mode=mode, normalize=normalize, as_frame=True)