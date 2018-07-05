import numpy as np

def generate(ts,
             fs,
             path,
             t_profile,
             f_profile,
             bp_profile):
    """Generates synthetic signal.

    Computes synethic signal using given path in time-frequency domain and
    brightness profiles in time and frequency directions.

    Parameters
    ----------
    ts : ndarray
        Time samples
    fs : ndarray
        Frequency samples
    path : function
        Function in time that returns frequencies
    t_profile : function
        Time profile: function in time that returns an intensity (scalar)
    f_profile : function
        Frequency profile: function in frequency that returns an intensity
        (scalar), relative to the signal frequency within a time sample
    bp_profile : function
        Bandpass profile: function in frequency that returns an intensity
        (scalar)

    Returns
    -------
    signal : ndarray
        Two-dimensional NumPy array containing synthetic signal data

    Examples
    --------
    A simple example that creates a linear Doppler-drifted signal:

    >>> import setigen as stg
    >>> import numpy as np
    >>> tsamp = 18.25361108
    >>> fch1 = 6095.214842353016
    >>> df = -2.7939677238464355e-06
    >>> fchans = 1024
    >>> tchans = 16
    >>> fs = np.arange(fch1, fch1 + fchans * df, df)
    >>> ts = np.arange(0, tchans * tsamp, tsamp)
    >>> signal = stg.generate(ts,
                              fs,
                              stg.constant_path(f_start = fs[200], drift_rate = -0.000002),
                              stg.constant_t_profile(level = 1),
                              stg.box_f_profile(width = 0.00001),
                              stg.constant_bp_profile(level = 1))

    The synthetic signal can then be visualized and saved within a Jupyter
    notebook using

    >>> %matplotlib inline
    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure(figsize=(10,6))
    >>> plt.imshow(signal, aspect='auto')
    >>> plt.colorbar()
    >>> fig.savefig("image.png", bbox_inches='tight')

    To run within a script, simply exclude the first line: :code:`%matplotlib inline`.

    """
    ff, tt = np.meshgrid(fs, ts)
    return t_profile(tt) * f_profile(ff, path(tt)) * bp_profile(ff)
