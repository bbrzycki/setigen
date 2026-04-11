import copy
import pickle

import numpy as np

from astropy import units as u
from astropy.time import Time
from astropy.stats import sigma_clip

from . import unit_utils
from . import slice
from . import plots
from . import utils
from ._frame.construction import (
    _attach_loaded_waterfall,
    _initialize_frame_from_spec,
    _normalize_frame_init,
)
from ._frame.models import (
    _ConstantSignalConfig,
    _NoiseConfig,
    _SampledNoiseConfig,
    _build_constant_signal_kwargs,
    _generate_noise,
    _generate_sampled_noise,
)
from ._frame.io import (
    _decode_bytestrings,
    _encode_bytestrings,
    _update_waterfall,
)
from ._frame.signal import (
    _finalize_signal,
    _get_restricted_fs,
    _normalize_bp_profile,
    _normalize_path,
    _normalize_t_profile,
    _render_signal,
    _resolve_bounding_indices,
)

class Frame(object):
    """
    A class to facilitate the creation of entirely synthetic radio data 
    (narrowband signals + background noise) as well as signal injection into 
    existing observations.
    """
    def __init__(self,
                 waterfall=None,
                 fchans=None,
                 tchans=None,
                 df=2.7939677238464355*u.Hz,
                 dt=18.253611008*u.s,
                 fch1=6*u.GHz,
                 ascending=False,
                 data=None,
                 seed=None,
                 **kwargs):
        """
        Initialize a Frame object either from an existing .fil/.h5 file or
        from frame resolution / size.

        If you are initializing based on a .fil or .h5, pass in either the
        filename or the Waterfall object into the waterfall keyword.

        Otherwise, you can initialize a frame by specifying the parameters
        ``fchans``, ``tchans``, ``df``, ``dt``, and even ``fch1``, if it's important to
        specify frequencies (8 GHz is an arbitrary but reasonable choice
        otherwise). Note that the frame resolutions ``df`` and ``dt`` are given 
        defaults based on the Breakthrough Listen high frequency resolution
        data product -- be sure to change these if you are working with 
        different kinds of data.
        
        The `data` keyword is only necessary if you are also
        preloading data that matches your specified frame dimensions and
        resolutions.

        Parameters
        ----------
        waterfall : str or Waterfall, optional
            Name of filterbank file or Waterfall object for preloading data
        fchans : int, optional
            Number of frequency samples
        tchans: int, optional
            Number of time samples
        df : astropy.Quantity, optional
            Frequency resolution (e.g. in u.Hz)
        dt : astropy.Quantity, optional
            Time resolution (e.g. in u.s)
        fch1 : astropy.Quantity, optional
            Central frequency of first channel. If ``ascending=True``, 
            ``fch1`` is the minimum frequency; if ``ascending=False`` 
            (default), ``fch1`` is the maximum frequency.
        ascending : bool, optional
            Specify whether frequencies should be in ascending order, so that 
            ``fch1`` is the minimum frequency. Default is False, for which ``fch1``
            is the maximum frequency. This is overwritten if a waterfall
            object is provided, where ``ascending`` will be automatically 
            determined by observational parameters.
        data : ndarray, optional
            2D array of intensities to preload into frame
        seed : None, int, Generator, optional
            Random seed or seed generator
        **kwargs
            For convenience, the ``shape`` keyword can be used in place of individually
            setting ``fchans`` and ``tchans``, so that ``shape=(tchans, fchans)``.
        """
        self.rng = np.random.default_rng(seed)
        _initialize_frame_from_spec(
            self,
            _normalize_frame_init(waterfall=waterfall,
                                  fchans=fchans,
                                  tchans=tchans,
                                  df=df,
                                  dt=dt,
                                  fch1=fch1,
                                  ascending=ascending,
                                  data=data,
                                  kwargs=kwargs),
        )
            
        # Degrees of freedom for chi-squared radiometer noise
        # 2 polarizations, real and imaginary components -> 4
        self.chi2_df = 4 * round(self.df * self.dt)
        
        # Calculate unit drift rate (pixel over pixel drift)
        self.unit_drift_rate = self.df / self.dt

        # Shared creation of ranges
        self._update_fs()
        self._update_ts()

        # No matter what, self.data will be populated at this point.
        self._update_noise_frame_stats()

        # Placeholder dictionary for user metadata, just for bookkeeping purposes
        self.metadata = self.get_params()

    @classmethod
    def from_data(cls, df, dt, fch1, ascending, data, metadata=None, waterfall=None, seed=None):
        """
        Initialize Frame more directly from 2D numpy array of data.
        
        Parameters
        ----------
        df : astropy.Quantity
            Frequency resolution (e.g. in u.Hz)
        dt : astropy.Quantity
            Time resolution (e.g. in u.s)
        fch1 : astropy.Quantity
            Central frequency of first channel. If ``ascending=True``, 
            ``fch1`` is the minimum frequency; if ``ascending=False`` 
            (default), ``fch1`` is the maximum frequency.
        ascending : bool
            Specify whether frequencies should be in ascending order, so that 
            ``fch1`` is the minimum frequency. Default is False, for which ``fch1``
            is the maximum frequency. This is overwritten if a waterfall
            object is provided, where ``ascending`` will be automatically 
            determined by observational parameters.
        data : ndarray
            2D array of intensities to preload into frame
        metadata : dict, optional
            Dictionary of features associated with the frame
        waterfall : Waterfall, optional
            Associated Waterfall object if data is derived from another frame object 
            (accessed via ``frame.get_waterfall()``) or a blimpy waterfall object
        seed : None, int, Generator, optional
            Random seed or seed generator
            
        Returns
        -------
        frame : Frame
            Frame object with preloaded data
        """
        tchans, fchans = data.shape
        frame = cls(fchans=fchans,
                    tchans=tchans,
                    df=df,
                    dt=dt,
                    fch1=fch1,
                    ascending=ascending,
                    data=data,
                    seed=seed)
        if metadata is not None:
            frame.add_metadata(dict(metadata))

        _attach_loaded_waterfall(frame, waterfall)
        return frame

    @classmethod
    def from_waterfall(cls, waterfall, seed=None):
        """
        Instantiate Frame using a filterbank file or blimpy Waterfall object.
        """
        return cls(waterfall=waterfall, seed=seed)
    
    @classmethod
    def from_backend_params(cls,
                            fchans=None,
                            obs_length=300, 
                            sample_rate=3e9, 
                            num_branches=1024,
                            fftlength=1048576,
                            int_factor=51,
                            fch1=6*u.GHz,
                            ascending=False,
                            data=None,
                            seed=None):
        """
        Create frame based on backend / software related parameters.
        Either ``fchans`` or ``data`` must be provided to get number of frequency
        channels to create. If a 2D numpy array for ``data`` is provided, ``fchans``
        will be inferred. The parameter ``int_factor`` must still be provided 
        to determine ``tchans``; there is a check that the data dimensions also match.
        Since multiple ``int_factor`` values may correspond to the same ``tchans``, 
        for clarity we do not infer ``int_factor`` just from the dimensions of the data.
        
        Parameters
        ----------
        fchans : int, optional
            Number of frequency samples. Should be provided if ``data`` is None.
        obs_length : float, optional
            Length of observation in seconds
        sample_rate : float, optional
            Physical sample rate, in Hz, for collecting real voltage data
        num_branches : int, optional
            Number of PFB branches. Note that this corresponds to ``num_branches / 2`` coarse channels.
        fftlength : int, optional
            FFT length to be used in fine channelization
        int_factor : int, optional
            Integration factor used in fine channelization. Determines ``tchans``.
        fch1 : astropy.Quantity, optional
            Central frequency of first channel. If ``ascending=True``, 
            ``fch1`` is the minimum frequency; if ``ascending=False`` 
            (default), ``fch1`` is the maximum frequency.
        ascending : bool, optional
            Specify whether frequencies should be in ascending order, so that 
            ``fch1`` is the minimum frequency. Default is False, for which ``fch1``
            is the maximum frequency.
        data : ndarray, optional
            2D array of intensities to preload into frame. If provided, ``fchans``
            will be inferred from this. 
        seed : None, int, Generator, optional
            Random seed or seed generator
            
        Returns
        -------
        frame : Frame
            Frame object with appropriate dimensions.
        """
        chan_bw = sample_rate / num_branches
        df = chan_bw / fftlength
        
        if data is not None:
            tchans, fchans = data.shape
        elif fchans is None:
            raise ValueError("Value not given for fchans")
            
        param_dict = params_from_backend(obs_length=obs_length,
                                         sample_rate=sample_rate,
                                         num_branches=num_branches,
                                         fftlength=fftlength,
                                         int_factor=int_factor)
        if data is not None:
            if param_dict['tchans'] != tchans:
                raise ValueError(
                    f"Data has {tchans} time samples, but backend parameters imply {param_dict['tchans']}."
                )
        
        frame = cls(fchans=fchans,
                    **param_dict,
                    fch1=fch1,
                    ascending=ascending,
                    data=data,
                    seed=seed)
        return frame
        
    def copy(self):
        """
        Return identical copy of frame.
        """
        c_frame = copy.deepcopy(self)
        # Note that since the __getstate__ function is overwritten, we need to
        # add back the waterfall object.
        waterfall = self.get_waterfall()
        if waterfall is not None:
            c_frame.waterfall = copy.deepcopy(waterfall)
        return c_frame

    def __getstate__(self):
        # Exclude waterfall Waterfall object from pickle, since it uses open threads, which
        # can't be pickled -- note that this affects copy!
        state = self.__dict__.copy()
        state['waterfall'] = None
        return state

    def _update_fs(self):
        """
        Calculate and update an array of frequencies represented in the
        frame.
        """
        # Normally, self.ascending will be False; filterbank convention is decreasing freqs
        if self.ascending:
            self.fmin = self.fch1
            self.fs = np.linspace(self.fmin,
                                  self.fmin + self.fchans * self.df,
                                  self.fchans,
                                  endpoint=False)
            self.fmax = self.fs[-1]
        else:
            self.fmax = self.fch1
            self.fs = np.linspace(self.fmax,
                                  self.fmax - self.fchans * self.df,
                                  self.fchans,
                                  endpoint=False)
            self.fmin = self.fs[-1]
            self.fs = self.fs[::-1]

    def _update_ts(self):
        """
        Calculate and update an array of times represented in the frame.
        """
        self.ts = unit_utils.get_value(np.linspace(0,
                                                   self.tchans * self.dt,
                                                   self.tchans,
                                                   endpoint=False),
                                       u.s)

    @property
    def fmid(self):
        return (self.fmin + self.fmax) / 2
        
    @property
    def mjd(self):
        return Time(self.t_start, format='unix').mjd
    
    @property
    def t_stop(self):
        return self.t_start + self.tchans * self.dt

    @property
    def obs_length(self):
        return self.tchans * self.dt
    
    @property 
    def ts_ext(self):
        """
        Extended time array of length ``tchans + 1``, including the ending
        timestamp.    
        """
        return np.append(self.ts, self.ts[-1] + self.dt)

    @property
    def mean(self):
        return np.mean(self.data)

    @property
    def std(self):
        return np.std(self.data)

    def get_total_stats(self):
        return self.mean, self.std

    def get_noise_stats(self):
        return self.noise_mean, self.noise_std

    def _update_noise_frame_stats(self):
        """
        Calculate and update basic noise statistics (mean and standard
        deviation) of the frame, using sigma clipping to strip outliers.
        """
        clipped_data = sigma_clip(self.data,
                                  sigma=3,
                                  maxiters=5,
                                  masked=False)
        self.noise_mean = np.mean(clipped_data)
        self.noise_std = np.std(clipped_data)

    def zero_data(self):
        """
        Reset data to a numpy array of zeros.
        """
        self.data = np.zeros(self.shape)
        self.noise_mean = self.noise_std = 0

    def add_noise(self,
                  x_mean,
                  x_std=None,
                  x_min=None,
                  noise_type='chi2'):
        """
        By default, synthesize radiometer noise based on a chi-squared
        distribution. Alternately, can generate pure Gaussian noise.
        
        Specifying ``noise_type='chi2'`` will only use ``x_mean``,
        and ignore other parameters. Specifying ``noise_type='gaussian'``
        will use all arguments (if provided).
        
        When adding Gaussian noise to the frame, the minimum is simply a
        lower bound for intensities in the data (e.g. it may make sense to
        cap intensities at 0), but this is optional.

        Parameters
        ----------
        x_mean : float
            Target mean
        x_std : float, optional
            Target standard deviation
        x_min : float, optional
            Lower bound for Gaussian noise
        noise_type : {"chi2", "gaussian", "normal"}, default: "chi2"
            Distribution to use for synthetic noise

        Return
        ------
        noise : ndarray
            Array of synthetic noise
        """
        noise, x_mean, x_std = _generate_noise(
            _NoiseConfig.from_values(x_mean=x_mean,
                                     x_std=x_std,
                                     x_min=x_min,
                                     noise_type=noise_type),
            chi2_df=self.chi2_df,
            shape=self.shape,
            rng=self.rng,
        )
                
        self.data += noise

        set_to_param = (self.noise_mean == self.noise_std == 0)
        if set_to_param:
            self.noise_mean, self.noise_std = x_mean, x_std
        else:
            self._update_noise_frame_stats()

        return noise

    def add_noise_from_obs(self,
                           x_mean_array=None,
                           x_std_array=None,
                           x_min_array=None,
                           share_index=True,
                           noise_type='chi2'):
        """
        By default, synthesize radiometer noise based on a chi-squared
        distribution. Alternately, can generate pure Gaussian noise.
        
        If no arrays are specified from which to sample, noise
        samples will be drawn from saved GBT C-Band observations at
        (dt, df) = (1.4 s, 1.4 Hz) resolution, from frames of shape
        (tchans, fchans) = (32, 1024). These sample noise parameters consist
        of 126500 samples for mean, std, and min of each observation.
        
        Specifying ``noise_type='chi2'`` will only use ``x_mean_array`` (if provided),
        and ignore other parameters. Specifying noise_type='gaussian' will use
        all arrays (if provided).

        Note: this method will attempt to scale the noise parameters to match
        self.dt and self.df. This assumes that the observation data products
        are *not* normalized by the FFT length used to construct them.

        Parameters
        ----------
        x_mean_array : ndarray, optional
            Array of potential means
        x_std_array : ndarray, optional
            Array of potential standard deviations
        x_min_array : ndarray, optional
            Array of potential minimum values
        share_index : bool, optional, default: True
            Whether to select noise parameters from the same index across each
            provided array. If True, then each array must be the same length.
        noise_type : {"chi2", "gaussian", "normal"}, default: "chi2"
            Distribution to use for synthetic noise

        Return
        ------
        noise : ndarray
            Array of synthetic noise
        """
        noise, x_mean, x_std = _generate_sampled_noise(
            _SampledNoiseConfig.from_values(x_mean_array=x_mean_array,
                                            x_std_array=x_std_array,
                                            x_min_array=x_min_array,
                                            share_index=share_index,
                                            noise_type=noise_type),
            dt=self.dt,
            chi2_df=self.chi2_df,
            shape=self.shape,
            rng=self.rng,
        )

        self.data += noise

        set_to_param = (self.noise_mean == self.noise_std == 0)
        if set_to_param:
            self.noise_mean, self.noise_std = x_mean, x_std
        else:
            self._update_noise_frame_stats()

        return noise

    def add_signal(self,
                   path,
                   t_profile,
                   f_profile,
                   bp_profile=None,
                   bounding_f_range=None,
                   integrate_path=False,
                   integrate_t_profile=False,
                   integrate_f_profile=False,
                   doppler_smearing=False,
                   t_subsamples=10,
                   f_subsamples=10,
                   smearing_subsamples=10):
        """
        Generate synthetic signal.

        Add a synethic signal using given path in time-frequency domain and
        brightness profiles in time and frequency directions.

        Parameters
        ----------
        path : function, np.ndarray, list, float
            Function in time that returns frequencies, or provided array or
            single value of frequencies for the center of the signal at each
            time sample
        t_profile : function, np.ndarray, list, float
            Time profile: function in time that returns an intensity (scalar),
            or provided array or single value of intensities at each time
            sample
        f_profile : function
            Frequency profile: function in frequency that returns an intensity
            (scalar), relative to the signal frequency within a time sample.
            Note that unlike the other parameters, this must be a function
        bp_profile : function, np.ndarray, list, float, optional
            Bandpass profile: function in frequency that returns a relative
            intensity (scalar, between 0 and 1), or provided array or single
            value of relative intensities at each frequency sample
        bounding_f_range : tuple
            Tuple (bounding_min, bounding_max) that constrains the computation
            of the signal to only a range in frequencies
        integrate_path : bool, optional
            Option to average path along time to get a more accurate frequency
            position in t-f space. Note that this option only makes sense if
            the provided path can be evaluated at the sub frequency sample
            level (e.g. as opposed to returning a pre-computed array of
            frequencies of length ``tchans``). Makes ``t_subsamples`` calculations
            per time sample.
        integrate_t_profile : bool, optional
            Option to integrate ``t_profile`` in the time direction. Note that
            this option only makes sense if the provided ``t_profile`` can be
            evaluated at the sub time sample level (e.g. as opposed to
            returning an array of intensities of length ``tchans``). Makes
            ``t_subsamples`` calculations per time sample.
        integrate_f_profile : bool, optional
            Option to integrate ``f_profile`` in the frequency direction. Makes
            ``f_subsamples`` calculations per time sample.
        doppler_smearing : bool, optional
            Option to numerically "Doppler smear" spectral power over 
            frequency bins. At time t, averages ``smearing_subsamples`` copies of
            the signal centered at evenly spaced center frequencies between 
            times t and t+1. This causes the effective drop in power when 
            the signal crosses multiple bins.
        t_subsamples : int, optional
            Number of bins for integration in the time direction, using
            Riemann sums. Default is 10.
        f_subsamples : int, optional
            Number of bins for integration in the frequency direction, using
            Riemann sums. Default is 10.
        smearing_subsamples : int, optional
            Number of steps for averaging evenly spaced copies of the signal 
            between center frequencies at times t and t+1. Default is 10.
        Returns
        -------
        signal : ndarray
            Two-dimensional NumPy array containing synthetic signal data

        Examples
        --------
        Here's an example that creates a linear Doppler-drifted signal with
        chi-squared noise with sampled parameters:

        >>> from astropy import units as u
        >>> import setigen as stg
        >>> fchans = 1024
        >>> tchans = 32
        >>> df = 2.7939677238464355*u.Hz
        >>> dt = tsamp = 18.253611008*u.s
        >>> fch1 = 6095.214842353016*u.MHz
        >>> frame = stg.Frame(fchans=fchans,
                              tchans=tchans,
                              df=df,
                              dt=dt,
                              fch1=fch1)
        >>> noise = frame.add_noise(x_mean=10)
        >>> signal = frame.add_signal(stg.constant_path(f_start=frame.get_frequency(200),
                                                        drift_rate=2*u.Hz/u.s),
                                      stg.constant_t_profile(level=frame.get_intensity(snr=30)),
                                      stg.gaussian_f_profile(width=40*u.Hz),
                                      stg.constant_bp_profile(level=1))

        Saving the noise and signals individually may be useful depending on
        the application, but the combined data can be accessed via
        frame.get_data(). The synthetic signal can then be visualized and
        saved within a Jupyter notebook using:

        >>> %matplotlib inline
        >>> import matplotlib.pyplot as plt
        >>> fig = plt.figure(figsize=(10, 6))
        >>> frame.plot()
        >>> plt.savefig('image.png', bbox_inches='tight')
        >>> plt.show()

        To run within a script, simply exclude the first line:
        ``%matplotlib inline``.

        """
        bounding_min, bounding_max = _resolve_bounding_indices(self, bounding_f_range)

        restricted_fs, restricted_fchans = _get_restricted_fs(
            self,
            bounding_min=bounding_min,
            bounding_max=bounding_max,
            integrate_f_profile=integrate_f_profile,
            f_subsamples=f_subsamples,
        )
        ff, _ = np.meshgrid(restricted_fs, self.ts)

        t_profile_tt = _normalize_t_profile(self,
                                            restricted_fs,
                                            t_profile,
                                            integrate_t_profile=integrate_t_profile,
                                            t_subsamples=t_subsamples)

        resolved_path = _normalize_path(self,
                                        restricted_fs,
                                        path,
                                        integrate_path=integrate_path,
                                        doppler_smearing=doppler_smearing,
                                        t_subsamples=t_subsamples,
                                        smearing_subsamples=smearing_subsamples)

        bp_profile_ff = _normalize_bp_profile(self, restricted_fs, bp_profile)

        signal = _render_signal(ff=ff,
                                t_profile_tt=t_profile_tt,
                                f_profile=f_profile,
                                bp_profile_ff=bp_profile_ff,
                                path_tt=resolved_path.path_tt,
                                doppler_smearing=doppler_smearing,
                                dpath_tt=resolved_path.dpath_tt,
                                smearing_subsamples=smearing_subsamples)

        return _finalize_signal(self,
                                signal=signal,
                                bounding_min=bounding_min,
                                bounding_max=bounding_max,
                                integrate_f_profile=integrate_f_profile,
                                restricted_fchans=restricted_fchans,
                                f_subsamples=f_subsamples)

    def add_constant_signal(self,
                            f_start,
                            drift_rate,
                            level,
                            width,
                            f_profile_type='sinc2',
                            doppler_smearing=False):
        """
        A wrapper around add_signal() that injects a constant intensity,
        constant drift_rate signal into the frame.

        Parameters
        ----------
        f_start : astropy.Quantity
            Starting signal frequency
        drift_rate : astropy.Quantity
            Signal drift rate, in units of frequency per time
        level : float
            Signal intensity
        width : astropy.Quantity
            Signal width in frequency units
        f_profile_type : {"sinc2", "box", "gaussian", "lorentzian", "voigt}, default: "sinc2"
            Signal spectral profile
        doppler_smearing : bool, optional, default: False
            Option to numerically "Doppler smear" spectral power over 
            frequency bins. At time t, averages ``drift_rate / frame.unit_drift_rate`` 
            copies of the signal centered at evenly spaced center frequencies between 
            times t and t+1. This causes the effective drop in power when 
            the signal crosses multiple bins.

        Returns
        -------
        signal : ndarray
            Two-dimensional NumPy array containing synthetic signal data
        """
        f_start = unit_utils.get_value(f_start, u.Hz)
        drift_rate = unit_utils.get_value(drift_rate, u.Hz / u.s)
        width = unit_utils.get_value(width, u.Hz)

        return self.add_signal(**_build_constant_signal_kwargs(
            self,
            _ConstantSignalConfig.from_values(f_start=f_start,
                                              drift_rate=drift_rate,
                                              level=level,
                                              width=width,
                                              f_profile_type=f_profile_type,
                                              doppler_smearing=doppler_smearing),
        ))

    def get_index(self, frequency):
        """
        Convert frequency to closest index in frame.
        """
        return np.round((unit_utils.get_value(frequency, u.Hz) - self.fmin) / self.df).astype(int)

    def get_frequency(self, index):
        """
        Convert index to frequency.
        """
        return self.fmin + self.df * index

    def get_intensity(self, snr):
        """
        Calculate intensity from SNR, based on estimates of the noise in the
        frame.

        Note that there must be noise present in the frame for this to make
        sense.
        """
        if self.noise_std == 0:
            raise ValueError('You must add noise in the image to specify SNR!')
        return snr * self.noise_std / np.sqrt(self.tchans)

    def get_snr(self, intensity):
        """
        Calculate SNR from intensity.

        Note that there must be noise present in the frame for this to make
        sense.
        """
        if self.noise_std == 0:
            raise ValueError('You must add noise in the image to return SNR!')
        return intensity * np.sqrt(self.tchans) / self.noise_std

    def get_drift_rate(self, start_index, stop_index):
        return (stop_index - start_index) * self.df / (self.tchans * self.dt)

    def get_info(self):
        return vars(self)
    
    def get_params(self):
        return {
            'fchans': self.fchans,
            'tchans': self.tchans,
            'df': self.df,
            'dt': self.dt,
            'fch1': self.fch1,
            'ascending': self.ascending
        }

    def get_data(self, db=False):
        if db:
            return 10 * np.log10(self.data)
        return self.data

    def get_metadata(self):
        return self.metadata

    def add_metadata(self, new_metadata):
        """
        Append custom metadata using a dictionary new_metadata.
        """
        self.metadata.update(new_metadata)
        
    def update_metadata(self, new_metadata):
        self.add_metadata(new_metadata)
        
    @utils._copy_docstring(plots.plot_frame)
    def plot(self, *args, **kwargs):
        return plots.plot_frame(self, *args, **kwargs)
        
    @utils._copy_docstring(slice.get_slice)
    def get_slice(self, *args, **kwargs):
        return slice.get_slice(self, *args, **kwargs)
        
    # @utils._copy_docstring(frame_utils.integrate)
    # def integrate(self, *args, **kwargs):
    #     return frame_utils.integrate(self, *args, **kwargs)
        
    def get_waterfall(self):
        """
        Return current frame as a Waterfall object. Note: some filterbank
        metadata may not be accurate anymore, depending on prior frame
        manipulations.
        """
        _update_waterfall(self)
        return self.waterfall
    
    def check_waterfall(self):
        """
        If an associated Waterfall object exists, update and return it. Otherwise,
        return None. Useful to chain with ``setigen.Frame.from_data()`` if manipulating
        completely synthetic data.
        """
        if self.waterfall is None:
            return None
        return self.get_waterfall()

    def save_fil(self, filename, max_load=1):
        """
        Save frame data as a filterbank file (.fil).
        """
        _update_waterfall(self, filename=filename, max_load=max_load)
        _encode_bytestrings(self)
        self.waterfall.write_to_fil(filename)
        _decode_bytestrings(self)

    def save_hdf5(self, filename, max_load=1):
        """
        Save frame data as an HDF5 file.
        """
        _update_waterfall(self, filename=filename, max_load=max_load)
        _encode_bytestrings(self)
        self.waterfall.write_to_hdf5(filename)
        _decode_bytestrings(self)

    def save_h5(self, filename, max_load=1):
        """
        Save frame data as an HDF5 file.
        """
        self.save_hdf5(filename, max_load=max_load)

    def save_npy(self, filename):
        """
        Save frame data as an .npy file.
        """
        np.save(filename, self.data)

    def load_npy(self, filename):
        """
        Load frame data from a .npy file.
        """
        self.data = np.load(filename)

    def save_pickle(self, filename):
        """
        Save entire frame as a pickled file (.pickle).
        """
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_pickle(cls, filename):
        """
        Load Frame object from a pickled file (.pickle), created with 
        :func:`~setigen.frame.Frame.save_pickle`.
        """
        with open(filename, "rb") as f:
            return pickle.load(f)

    
def params_from_backend(obs_length=300, 
                        sample_rate=3e9, 
                        num_branches=1024,
                        fftlength=1048576,
                        int_factor=51):
    """
    Return frame parameters calculated from data backend characteristics.

    Parameters
    ----------
    obs_length : float, optional
        Length of observation in seconds
    sample_rate : float, optional
        Physical sample rate, in Hz, for collecting real voltage data
    num_branches : int, optional
        Number of PFB branches. Note that this corresponds to 
        ``num_branches / 2`` coarse channels.
    fftlength : int, optional
        FFT length to be used in fine channelization
    int_factor : int, optional
        Integration factor used in fine channelization. Determines ``tchans``.

    Returns
    -------
    param_dict : dict
        Dictionary of parameters
    """
    chan_bw = sample_rate / num_branches
    df = chan_bw / fftlength

    dt = int_factor / df
    tchans = int(obs_length / dt)

    return {
        'tchans': tchans,
        'df': df,
        'dt': dt
    }
