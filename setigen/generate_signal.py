import numpy as np
import scipy.integrate as sciintegrate
import scipy.special as special

def generate(ts,
             fs,
             path,
             t_profile,
             f_profile,
             bp_profile,
             integrate = False,
             samples = 10):
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
    integrate : bool, optional
        Option to integrate t_profile in the time direction
    samples : int, optional
        Number of bins to integrate t_profile in the time direction, using
        Riemann sums

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
    # Assuming len(ts) >= 2
    tsamp = ts[1] - ts[0]
    ff, tt = np.meshgrid(fs, ts - tsamp / 2.)

    if integrate:
        new_ts = np.arange(0, ts[-1] + tsamp, tsamp / samples)
        y = t_profile(new_ts)
        if type(y) != np.ndarray:
            y = np.repeat(y, len(new_ts))
        new_y = []
        for i in range(len(ts)):
            tot = 0
            for j in range(samples*i, samples*(i+1)):
                tot += y[j]
            new_y.append(tot / samples)
        tt_profile = np.meshgrid(fs, new_y)[1]
    else:
        tt_profile = t_profile(tt)

    int_ts_path = []
    for i in range(len(ts)):
        val = sciintegrate.quad(path, ts[i], ts[i] + tsamp, limit=10)[0] / tsamp
        int_ts_path.append(val)
    int_tt_path = np.meshgrid(fs, int_ts_path)[1]

    signal = tt_profile * f_profile(ff, int_tt_path) * bp_profile(ff)
    return signal
