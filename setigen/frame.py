import os.path
import numpy as np
import scipy.integrate as sciintegrate
import scipy.special as special
from astropy import units as u

from blimpy import Waterfall

from . import fil_utils
from . import distributions
from . import sample_from_obs
from . import unit_utils


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
        if not None in [fchans, tchans, df, dt, fch1]:
            self.fil = None
            
            # Need to address this and come up with a meaningful header
            self.header = None
            self.fchans = int(unit_utils.get_value(fchans, u.pixel))
            self.df = unit_utils.get_value(df, u.Hz)
            self.fch1 = unit_utils.get_value(fch1, u.Hz)
            
            self.tchans = int(unit_utils.get_value(tchans, u.pixel))
            self.dt = unit_utils.get_value(dt, u.s)
            
            self.shape = (self.tchans, self.fchans)
            
            if data is not None:
                assert data.shape == self.shape
                self.data = data
            else:
                self.data = np.zeros(self.shape)
        elif fil:
            self.fil = fil
            self.header = fil.header
            self.fchans = fil.header[b'nchans']
            self.df = unit_utils.get_value(fil.header[b'foff'], u.Hz)
            self.fch1 = unit_utils.cast_value(fil.header[b'fch1'], u.MHz).to(u.Hz).value
            
            # When multiple Stokes parameters are supported, this will have to be expanded.
            self.data = fil_utils.get_data(fil)
            
            self.tchans = self.data.shape[0]
            self.dt = unit_utils.get_value(self.ts[1] - self.ts[0], u.s)
            
            self.shape = (self.tchans, self.fchans)
        else:
            raise ValueError('Frame must be provided dimensions or an existing filterbank file.')
            
        # Shared creation of ranges
        self.fs = unit_utils.get_value(np.arange(self.fch1, 
                                                 self.fch1 + self.fchans * self.df, 
                                                 self.df), 
                                       u.Hz)
        self.ts = unit_utils.get_value(np.arange(0, 
                                                 self.tchans * self.dt,
                                                 self.dt),
                                       u.s)
        
        # No matter what, self.data will be populated at this point.
        self._update_total_frame_stats()
        self._update_noise_frame_stats(exclude=0.1)
        
        
    def zero_data(self):
        self.data = np.zeros(self.shape)
        
        
    def _get_mean(self, exclude=0):
        flat_data = self.data.flatten()
        excluded_flat_data = np.sort(flat_data)[::-1][int(exclude * len(flat_data)):]
        return np.mean(excluded_flat_data)
    
    
    def _get_std(self, exclude=0):
        flat_data = self.data.flatten()
        excluded_flat_data = np.sort(flat_data)[::-1][int(exclude * len(flat_data)):]
        return np.std(excluded_flat_data)
    
    
    def _get_min(self, exclude=0):
        flat_data = self.data.flatten()
        excluded_flat_data = np.sort(flat_data)[::-1][int(exclude * len(flat_data)):]
        return np.min(excluded_flat_data)
    
    
    def _compute_frame_stats(self, exclude=0):
        flat_data = self.data.flatten()
        excluded_flat_data = np.sort(flat_data)[::-1][int(exclude * len(flat_data)):]
        
        frame_mean = np.mean(excluded_flat_data)
        frame_std = np.std(excluded_flat_data)
        frame_min = np.min(excluded_flat_data)
        
        return frame_mean, frame_std, frame_min
    
    
    def get_total_stats(self):
        return self.mean, self.std, self.min
    
    
    def get_noise_stats(self):
        return self.noise_mean, self.noise_std, self.noise_min
    
    
    def _update_total_frame_stats(self):
        self.mean, self.std, self.min = self._compute_frame_stats()
        
        
    def _update_noise_frame_stats(self, exclude=0.1):
        self.noise_mean, self.noise_std, self.noise_min = self._compute_frame_stats(exclude=exclude)
        
    
    def add_noise(self, 
                  x_mean,
                  x_std,
                  x_min=None):
        if x_min is not None:
            noise = distributions.truncated_gaussian(x_mean, 
                                                     x_std,
                                                     x_min, 
                                                     self.data.shape)
        else:
            noise = distributions.gaussian(x_mean,
                                           x_std, 
                                           self.data.shape)
        self.data += noise
        
        set_to_param = (self.mean == self.std == self.min == 0)
        self._update_total_frame_stats()
        self._update_noise_frame_stats(exclude=0.1)
        if set_to_param:
            self.noise_mean, self.noise_std = x_mean, x_std
        
        return noise
        
    
    def add_noise_from_obs(self,
                           x_mean_array=None,
                           x_std_array=None,
                           x_min_array=None):
        """
        If no arrays are specified to sample Gaussian parameters from, noise samples will be drawn from saved GBT C-Band observations at (dt, df) = (1.4 s, 1.4 Hz) resolution, from frames of shape (tchans, fchans) = (32, 1024). These sample noise parameters consists of 126500 samples for mean, std, and min of each observation.
        """
        if (x_mean_array is None and 
            x_std_array is None and 
            x_min_array is None):
            my_path = os.path.abspath(os.path.dirname(__file__))
            path = os.path.join(my_path, 'assets/sample_noise_params.npy')
            sample_noise_params = np.load(path)
            x_mean_array = sample_noise_params[:, 0]
            x_std_array = sample_noise_params[:, 1]
            x_min_array = sample_noise_params[:, 2]
        if x_min_array is not None:
            x_mean, x_std, x_min = sample_from_obs.sample_gaussian_params(x_mean_array,
                                                                          x_std_array,
                                                                          x_min_array)
            noise = distributions.truncated_gaussian(x_mean,
                                                     x_std,
                                                     x_min,
                                                     self.data.shape)
        else:
            x_mean, x_std = sample_from_obs.sample_gaussian_params(x_mean_array,
                                                                   x_std_array)
            noise = distributions.gaussian(x_mean,
                                           x_std,
                                           self.data.shape)
        self.data += noise
        
        set_to_param = (self.mean == self.std == self.min == 0)
        self._update_total_frame_stats()
        self._update_noise_frame_stats(exclude=0.1)
        if set_to_param:
            self.noise_mean, self.noise_std = x_mean, x_std
        
        return noise
    
    
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
        Here's an example that creates a linear Doppler-drifted signal with Gaussian noise with sampled parameters:

        >>> from astropy import units as u
        >>> import setigen as stg
        >>> fchans = 1024
        >>> tchans = 32
        >>> df = -2.7939677238464355*u.Hz
        >>> dt = tsamp = 18.25361108*u.s
        >>> fch1 = 6095.214842353016*u.MHz
        >>> frame = stg.Frame(fchans, tchans, df, dt, fch1)
        >>> noise = frame.add_noise(x_mean=5, x_std=2, x_min=0)
        >>> signal = frame.add_signal(stg.constant_path(f_start=frame.fs[200], drift_rate=-2*u.Hz/u.s),
                                      stg.constant_t_profile(level=frame.compute_intensity(snr=30)),
                                      stg.gaussian_f_profile(width=20*u.Hz),
                                      stg.constant_bp_profile(level=1))

        Saving the noise and signals individually may be useful depending on the application, but the combined data can be accessed via frame.get_data(). The synthetic signal can then be visualized and saved within a Jupyter notebook using:

        >>> %matplotlib inline
        >>> import matplotlib.pyplot as plt
        >>> fig = plt.figure(figsize=(10, 6))
        >>> plt.imshow(frame.get_data(), aspect='auto')
        >>> plt.xlabel('Frequency')
        >>> plt.ylabel('Time')
        >>> plt.colorbar()
        >>> plt.savefig('image.png', bbox_inches='tight')
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
        
        self._update_total_frame_stats()
        
        return signal
    
    
    def compute_intensity(self, snr):
        '''Calculate intensity from SNR'''
        if self.noise_std == 0:
            raise ValueError('You must add noise in the image to specify SNR!')
        return snr * self.noise_std / np.sqrt(self.tchans)
    
    
    def compute_SNR(self, intensity):
        '''Calculate SNR from intensity'''
        if self.noise_std == 0:
            raise ValueError('You must add noise in the image to return SNR!')
        return intensity * np.sqrt(self.tchans) / self.noise_std
    
    
    def get_info(self):
        return vars(self)
    
    
    def get_data(self, db=False):
        if db:
            return 10 * np.log10(self.data)
        return self.data
    
    
    def set_df(self, df):
        self.df = df
        self.fs = unit_utils.get_value(np.arange(self.fch1, 
                                                 self.fch1 + self.fchans * self.df, 
                                                 self.df), 
                                       u.Hz)
        
    def set_dt(self, dt):
        self.dt = dt
        self.ts = unit_utils.get_value(np.arange(0, 
                                                 self.tchans * self.dt, 
                                                 self.dt), 
                                       u.s)
    
    
    def set_data(self, data):
        self.data = data
        self.shape = data.shape
        self.tchans, self.fchans = self.shape
        self.fs = unit_utils.get_value(np.arange(self.fch1, 
                                                 self.fch1 + self.fchans * self.df, 
                                                 self.df), 
                                       u.Hz)
        self.ts = unit_utils.get_value(np.arange(0, 
                                                 self.tchans * self.dt, 
                                                 self.dt), 
                                       u.s)
        
    def save_fil(self, filename):
        '''IMPORTANT: this does not overwrite fil metadata, only the data'''
        if self.fil is None:
            my_path = os.path.abspath(os.path.dirname(__file__))
            path = os.path.join(my_path, 'assets/sample.fil')
            self.fil = Waterfall(path)
        self.fil.data = self.data
        self.fil.write_to_fil(filename)
    
    
    def load_fil(self, fil):
        '''IMPORTANT: this does not import fil metadata, only the data'''
        self.fil = fil
        self.set_data(fil_utils.get_data(fil))
    
    
    def save_data(self, file):
        '''file can be a filename or a file handle of a npy file'''
        np.save(file, self.data)
    
    
    def load_data(self, file):
        '''file can be a filename or a file handle of a npy file'''
        self.set_data(np.load(file))