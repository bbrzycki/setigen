import numpy as np
import scipy.integrate as sciintegrate
import scipy.special as special

from fil_utils import get_ts, get_fs, get_data


class Frame(object):
    '''
    An individual frame object to both construct and add to existing data.
    '''
    
    def __init__(self, 
                 fchans=None,
                 tchans=None,
                 df=None,
                 dt=None,
                 fch1=None,
                 data=None,
                 fil=None):
        if fchans and tchans and df and dt and fch1:
            # Need to address this and come up with a meaningful header
            self.header = None
            self.fchans = fchans
            self.tchans = tchans
            self.df = df
            self.dt = dt
            self.fch1 = fch1
            self.fs = np.arange(fch1, fch1 + fchans * df, df)
            self.ts = np.arange(0, tchans * dt, dt)
            if data:
                assert data.shape == (tchans, fchans)
                self.data = data
            else:
                self.data = np.zeros((tchans, fchans))
        elif fil:
            self.fchans = fil.header[b'nchans']
            self.df = fil.header[b'foff']
            self.fch1 = fil.header[b'fch1']
            self.fs = np.arange(self.fch1, self.fch1 + self.fchans * self.df, self.df)
            
            # When multiple Stokes parameters are supported, this will have to be expanded.
            self.data = get_data(fil)
            
            self.ts = get_ts(fil)
            self.tchans = len(self.ts)
            self.dt = self.ts[1] - self.ts[0]
            self.header = fil.header
        else:
            raise ValueError('Frame must be provided dimensions or an existing filterbank file.')
        
    
    def add_noise(self):
        pass
    
    
    def add_signal(self,
                   path,
                   t_profile,
                   f_profile,
                   bp_profile,
                   integrate_time=False,
                   samples=10,
                   average_f_pos=False):
        """Generates synthetic signal.

        Adds a synethic signal using given path in time-frequency domain and
        brightness profiles in time and frequency directions.

        Parameters
        ----------
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
        integrate_time : bool, optional
            Option to integrate t_profile in the time direction
        samples : int, optional
            Number of bins to integrate t_profile in the time direction, using
            Riemann sums
        average_f_pos : bool, optional
            Option to average path along frequency to get better position in t-f
            space

        Returns
        -------
        signal : ndarray
            Two-dimensional NumPy array containing synthetic signal data

        Examples
        --------
        A simple example that creates a linear Doppler-drifted signal:

        >>> import setigen as stg
        >>> fchans = 1024
        >>> tchans = 32
        >>> df = -2.7939677238464355e-06
        >>> dt = tsamp = 18.25361108
        >>> fch1 = 6095.214842353016
        >>> frame = stg.Frame(fchans, tchans, df, dt, fch1)
        >>> signal = frame.add_signal(stg.constant_path(f_start = frame.fs[200], drift_rate = -0.000002),
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
        >>> plt.savefig("image.png", bbox_inches='tight')
        >>> plt.show()

        To run within a script, simply exclude the first line: :code:`%matplotlib inline`.

        """
        # Assuming len(ts) >= 2
        ff, tt = np.meshgrid(self.fs, self.ts - self.dt / 2.)

        # Integrate in time direction to capture temporal variations more accurately
        if integrate_time:
            new_ts = np.arange(0, self.ts[-1] + self.dt, self.dt / samples)
            y = t_profile(new_ts)
            if type(y) != np.ndarray:
                y = np.repeat(y, len(new_ts))
            new_y = []
            for i in range(len(self.ts)):
                tot = 0
                for j in range(samples*i, samples*(i+1)):
                    tot += y[j]
                new_y.append(tot / samples)
            tt_profile = np.meshgrid(self.fs, new_y)[1]
        else:
            tt_profile = t_profile(tt)

        # TODO: optimize with vectorization and array operations.
        # Average using integration to get a better position in frequency direction
        if average_f_pos:
            int_ts_path = []
            for i in range(len(self.ts)):
                val = sciintegrate.quad(path, self.ts[i], self.ts[i] + self.dt, limit=10)[0] / self.tsamp
                int_ts_path.append(val)
        else:
            int_ts_path = path(self.ts)
        path_f_pos = np.meshgrid(self.fs, int_ts_path)[1]

        signal = tt_profile * f_profile(ff, path_f_pos) * bp_profile(ff)
        
        self.data += signal
        return signal
    
    
    def save_fil(self):
        pass
    
    
    def load_fil(self):
        pass
    