from __future__ import annotations

import copy
import pickle
from typing import Any

import numpy as np

from astropy import units as u
from astropy.time import Time
from astropy.stats import sigma_clip

from . import unit_utils
from . import slice
from . import plots
from . import utils
from ._typing import BandpassProfileInput, FrequencyPathInput, FrequencyProfile, PathLike, SeedLike, TimeProfileInput
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
    _resolve_auto_bounding_range,
    _resolve_bounding_indices,
)

class Frame(object):
    """Represent synthetic or waterfall-backed SETI spectrogram data."""
    def __init__(self,
                 waterfall: Any = None,
                 fchans: int | None = None,
                 tchans: int | None = None,
                 df: Any = 2.7939677238464355*u.Hz,
                 dt: Any = 18.253611008*u.s,
                 fch1: Any = 6*u.GHz,
                 ascending: bool = False,
                 data: np.ndarray | None = None,
                 seed: SeedLike = None,
                 **kwargs: Any) -> None:
        """Initialize a frame from synthetic dimensions or existing data.

        Args:
            waterfall: Waterfall object or path to a `.fil` or `.h5` file.
            fchans: Number of frequency channels for synthetic initialization.
            tchans: Number of time channels for synthetic initialization.
            df: Frequency resolution.
            dt: Time resolution.
            fch1: Frequency of the first channel.
            ascending: Whether the frequency axis should be ascending.
            data: Optional preloaded frame data.
            seed: Random seed or generator.
            **kwargs: Additional construction keywords such as `shape`, `mjd`,
                `t_start`, `source_name`, `f_start`, or `f_stop`.
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
    def from_data(
        cls,
        df: Any,
        dt: Any,
        fch1: Any,
        ascending: bool,
        data: np.ndarray,
        metadata: dict[str, Any] | None = None,
        waterfall: Any = None,
        seed: SeedLike = None,
        t_start: float | None = None,
        source_name: str | None = None,
        header: dict[str, Any] | None = None,
    ) -> "Frame":
        """Build a frame directly from an in-memory data array.

        Args:
            df: Frequency resolution.
            dt: Time resolution.
            fch1: Frequency of the first channel.
            ascending: Whether the frequency axis is ascending.
            data: Preloaded frame data.
            metadata: Optional metadata to attach to the frame.
            waterfall: Optional associated waterfall object.
            seed: Random seed or generator.
            t_start: Optional Unix start time.
            source_name: Optional source name.
            header: Optional filterbank-style header metadata.

        Returns:
            Frame populated with the supplied data.
        """
        tchans, fchans = data.shape
        init_kwargs = {}
        if t_start is not None:
            init_kwargs["t_start"] = t_start
        if source_name is not None:
            init_kwargs["source_name"] = source_name
        frame = cls(fchans=fchans,
                    tchans=tchans,
                    df=df,
                    dt=dt,
                    fch1=fch1,
                    ascending=ascending,
                    data=data,
                    seed=seed,
                    **init_kwargs)
        if metadata is not None:
            frame.add_metadata(dict(metadata))
        if header is not None:
            frame.header = copy.deepcopy(header)

        _attach_loaded_waterfall(frame, waterfall)
        return frame

    @classmethod
    def from_waterfall(cls, waterfall: Any, seed: SeedLike = None) -> "Frame":
        """Build a frame from a waterfall-backed observation.

        Args:
            waterfall: Waterfall object or path to a supported file.
            seed: Random seed or generator.

        Returns:
            Frame loaded from the supplied waterfall.
        """
        return cls(waterfall=waterfall, seed=seed)
    
    @classmethod
    def from_backend_params(cls,
                            fchans: int | None = None,
                            obs_length: float = 300,
                            sample_rate: float = 3e9,
                            num_branches: int = 1024,
                            fftlength: int = 1048576,
                            int_factor: int = 51,
                            fch1: Any = 6*u.GHz,
                            ascending: bool = False,
                            data: np.ndarray | None = None,
                            seed: SeedLike = None) -> "Frame":
        """Build a frame from backend-like observing parameters.

        Args:
            fchans: Number of frequency channels. Required when `data` is not
                supplied.
            obs_length: Observation length in seconds.
            sample_rate: Real-voltage sample rate in Hz.
            num_branches: Number of PFB branches.
            fftlength: Fine-channel FFT length.
            int_factor: Fine-channel integration factor.
            fch1: Frequency of the first channel.
            ascending: Whether the frequency axis is ascending.
            data: Optional preloaded frame data.
            seed: Random seed or generator.

        Returns:
            Frame with dimensions implied by the backend parameters.

        Raises:
            ValueError: If neither `fchans` nor `data` is supplied, or if the
                supplied data shape is inconsistent with the backend parameters.
        """
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
        
    def copy(self) -> "Frame":
        """Return a deep copy of the frame.

        Returns:
            Independent frame copy.
        """
        c_frame = copy.deepcopy(self)
        # Since __getstate__ excludes transient Waterfall adapters, preserve an
        # already-created in-memory adapter when it is safe to copy. Never create
        # a Waterfall as a side effect of copying.
        if self.waterfall is not None:
            try:
                c_frame.waterfall = copy.deepcopy(self.waterfall)
            except Exception:
                c_frame.waterfall = None
        return c_frame

    def __getstate__(self) -> dict[str, Any]:
        # Exclude waterfall Waterfall object from pickle, since it uses open threads, which
        # can't be pickled -- note that this affects copy!
        state = self.__dict__.copy()
        state['waterfall'] = None
        return state

    def _update_fs(self) -> None:
        """Update the frame frequency axis and derived bounds."""
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

    def _update_ts(self) -> None:
        """Update the frame time axis."""
        self.ts = unit_utils.get_value(np.linspace(0,
                                                   self.tchans * self.dt,
                                                   self.tchans,
                                                   endpoint=False),
                                       u.s)

    @property
    def fmid(self) -> float:
        """Return the midpoint frequency of the frame."""
        return (self.fmin + self.fmax) / 2
        
    @property
    def mjd(self) -> float:
        """Return the frame start time in Modified Julian Date."""
        return Time(self.t_start, format='unix').mjd
    
    @property
    def t_stop(self) -> float:
        """Return the observation stop time in Unix seconds."""
        return self.t_start + self.tchans * self.dt

    @property
    def obs_length(self) -> float:
        """Return the total observation length in seconds."""
        return self.tchans * self.dt
    
    @property 
    def ts_ext(self) -> np.ndarray:
        """Return the time axis extended by one final endpoint sample."""
        return self.time_edges

    @property
    def frequency_centers(self) -> np.ndarray:
        """Return frequency-channel center coordinates in Hz."""
        return self.fs

    @property
    def frequency_edges(self) -> np.ndarray:
        """Return frequency-channel edge coordinates in Hz."""
        return np.linspace(self.fmin - self.df / 2,
                          self.fmax + self.df / 2,
                          self.fchans + 1)

    @property
    def time_starts(self) -> np.ndarray:
        """Return time-bin start coordinates in seconds."""
        return self.ts

    @property
    def time_centers(self) -> np.ndarray:
        """Return time-bin center coordinates in seconds."""
        return self.ts + self.dt / 2

    @property
    def time_edges(self) -> np.ndarray:
        """Return time-bin edge coordinates in seconds."""
        return np.linspace(0,
                          self.tchans * self.dt,
                          self.tchans + 1,
                          endpoint=True)

    @property
    def mean(self) -> float:
        """Return the mean intensity of the frame."""
        return np.mean(self.data)

    @property
    def std(self) -> float:
        """Return the standard deviation of the frame."""
        return np.std(self.data)

    def get_total_stats(self) -> tuple[float, float]:
        """Return the mean and standard deviation of the full frame.

        Returns:
            Mean and standard deviation of the frame data.
        """
        return self.mean, self.std

    def get_noise_stats(self) -> tuple[float, float]:
        """Return the sigma-clipped noise statistics for the frame.

        Returns:
            Sigma-clipped noise mean and standard deviation.
        """
        return self.noise_mean, self.noise_std

    def _update_noise_frame_stats(self) -> None:
        """Update sigma-clipped noise statistics for the frame."""
        clipped_data = sigma_clip(self.data,
                                  sigma=3,
                                  maxiters=5,
                                  masked=False)
        self.noise_mean = np.mean(clipped_data)
        self.noise_std = np.std(clipped_data)

    def zero_data(self) -> None:
        """Reset frame data and cached noise statistics to zero."""
        self.data = np.zeros(self.shape)
        self.noise_mean = self.noise_std = 0

    def add_noise(self,
                  x_mean: float,
                  x_std: float | None = None,
                  x_min: float | None = None,
                  noise_type: str = 'chi2') -> np.ndarray:
        """Add synthetic radiometer or Gaussian noise to the frame.

        Args:
            x_mean: Target mean intensity.
            x_std: Target standard deviation for Gaussian noise.
            x_min: Optional lower bound for truncated Gaussian noise.
            noise_type: Noise distribution selector.

        Returns:
            Synthetic noise array added to the frame.

        Raises:
            ValueError: If Gaussian noise is requested without `x_std`.
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
                           x_mean_array: np.ndarray | None = None,
                           x_std_array: np.ndarray | None = None,
                           x_min_array: np.ndarray | None = None,
                           share_index: bool = True,
                           noise_type: str = 'chi2') -> np.ndarray:
        """Add synthetic noise by sampling empirical observation parameters.

        Args:
            x_mean_array: Candidate noise means.
            x_std_array: Candidate noise standard deviations.
            x_min_array: Candidate truncated-Gaussian minima.
            share_index: Whether to sample correlated parameters by shared index.
            noise_type: Noise distribution selector.

        Returns:
            Synthetic noise array added to the frame.

        Raises:
            IndexError: If shared-index sampling is requested for mismatched
                parameter arrays.
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
                   path: FrequencyPathInput,
                   t_profile: TimeProfileInput,
                   f_profile: FrequencyProfile,
                   bp_profile: BandpassProfileInput | None = None,
                   bounding_f_range: tuple[Any, Any] | None = None,
                   integrate_path: bool = False,
                   integrate_t_profile: bool = False,
                   integrate_f_profile: bool = False,
                   doppler_smearing: bool = False,
                   t_subsamples: int = 10,
                   f_subsamples: int = 10,
                   smearing_subsamples: int = 10,
                   t_offset: float = 0,
                   auto_bounding: bool = False,
                   truncate_below: float | None = None) -> np.ndarray:
        """Add a synthetic signal to the frame.

        Args:
            path: Signal path in time-frequency space.
            t_profile: Time-intensity profile.
            f_profile: Frequency profile callable.
            bp_profile: Optional bandpass profile.
            bounding_f_range: Optional bounding frequency range for rendering.
            integrate_path: Whether to oversample and average the path in time.
            integrate_t_profile: Whether to oversample and average the time
                profile.
            integrate_f_profile: Whether to oversample and average the frequency
                profile.
            doppler_smearing: Whether to numerically smear power across
                frequency bins.
            t_subsamples: Number of time subsamples per bin.
            f_subsamples: Number of frequency subsamples per bin.
            smearing_subsamples: Number of substeps used for Doppler smearing.
            t_offset: Time offset applied when evaluating callable time profiles
                and paths. This is primarily used for cadence-level injections.
            auto_bounding: Whether to infer a conservative frequency bounding
                range for known built-in frequency profiles.
            truncate_below: Optional relative power cutoff for supported
                infinite-support profiles when `auto_bounding` is enabled.

        Returns:
            Two-dimensional signal array that was added to the frame.
        """
        if doppler_smearing and smearing_subsamples < 1:
            raise ValueError("smearing_subsamples must be at least 1 when doppler_smearing=True")

        if auto_bounding and bounding_f_range is None:
            bounding_f_range, path = _resolve_auto_bounding_range(
                self,
                path,
                f_profile,
                integrate_path=integrate_path,
                integrate_f_profile=integrate_f_profile,
                doppler_smearing=doppler_smearing,
                t_subsamples=t_subsamples,
                t_offset=t_offset,
                truncate_below=truncate_below,
            )

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
                                            t_subsamples=t_subsamples,
                                            t_offset=t_offset)

        resolved_path = _normalize_path(self,
                                        restricted_fs,
                                        path,
                                        integrate_path=integrate_path,
                                        doppler_smearing=doppler_smearing,
                                        t_subsamples=t_subsamples,
                                        smearing_subsamples=smearing_subsamples,
                                        t_offset=t_offset)

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
                            f_start: Any,
                            drift_rate: Any,
                            level: float,
                            width: Any,
                            f_profile_type: str = 'sinc2',
                            doppler_smearing: bool = False) -> np.ndarray:
        """Add a constant-intensity, constant-drift signal to the frame.

        Args:
            f_start: Starting signal frequency.
            drift_rate: Signal drift rate.
            level: Signal intensity.
            width: Signal width.
            f_profile_type: Spectral profile selector.
            doppler_smearing: Whether to numerically smear power across bins.

        Returns:
            Two-dimensional signal array that was added to the frame.
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

    def get_index(self, frequency: Any) -> np.ndarray:
        """Convert frequency to the closest channel index.

        Args:
            frequency: Frequency or array of frequencies to convert.

        Returns:
            Closest frame index or indices.
        """
        return np.round((unit_utils.get_value(frequency, u.Hz) - self.fmin) / self.df).astype(int)

    def get_frequency(self, index: int | np.ndarray) -> float | np.ndarray:
        """Convert a frame index into frequency.

        Args:
            index: Frame index or indices.

        Returns:
            Frequency value or array in Hz.
        """
        return self.fmin + self.df * index

    def get_intensity(self, snr: float) -> float:
        """Calculate signal intensity from SNR using the frame noise estimate.

        Args:
            snr: Desired signal-to-noise ratio.

        Returns:
            Signal intensity that corresponds to the requested SNR.

        Raises:
            ValueError: If the frame does not yet contain measurable noise.
        """
        if self.noise_std == 0:
            raise ValueError('You must add noise in the image to specify SNR!')
        return snr * self.noise_std / np.sqrt(self.tchans)

    def get_snr(self, intensity: float) -> float:
        """Calculate SNR from signal intensity using the frame noise estimate.

        Args:
            intensity: Signal intensity.

        Returns:
            Signal-to-noise ratio for the supplied intensity.

        Raises:
            ValueError: If the frame does not yet contain measurable noise.
        """
        if self.noise_std == 0:
            raise ValueError('You must add noise in the image to return SNR!')
        return intensity * np.sqrt(self.tchans) / self.noise_std

    def get_drift_rate(self,
                       start_index: int,
                       stop_index: int,
                       reference: str = "edges") -> float:
        """Calculate drift rate from pixel coordinates.

        Args:
            start_index: Starting frequency index.
            stop_index: Ending frequency index.
            reference: Time reference convention. ``"edges"`` preserves the
                historical convention that spans the full integrated
                observation length. ``"centers"`` uses the distance between the
                representative time labels of the first and last rows.

        Returns:
            Drift rate in Hz/s.
        """
        if reference in {"edges", "edge", "time_edges", "integration"}:
            duration = self.tchans * self.dt
        elif reference in {"centers", "center", "time_centers", "labels"}:
            if self.tchans < 2:
                raise ValueError("center-referenced drift rates require at least two time bins")
            duration = (self.tchans - 1) * self.dt
        else:
            raise ValueError("reference must be 'edges' or 'centers'")
        return (stop_index - start_index) * self.df / duration

    def get_info(self) -> dict[str, Any]:
        """Return the full frame attribute dictionary.

        Returns:
            Full frame attribute dictionary.
        """
        return vars(self)
    
    def get_params(self) -> dict[str, Any]:
        """Return the core frame parameters.

        Returns:
            Dictionary of primary frame parameters.
        """
        return {
            'fchans': self.fchans,
            'tchans': self.tchans,
            'df': self.df,
            'dt': self.dt,
            'fch1': self.fch1,
            'ascending': self.ascending
        }

    def get_data(self, db: bool = False) -> np.ndarray:
        """Return frame data in linear or decibel units.

        Args:
            db: Whether to convert intensities to dB before returning them.

        Returns:
            Frame data array.
        """
        if db:
            return 10 * np.log10(self.data)
        return self.data

    def get_metadata(self) -> dict[str, Any]:
        """Return attached frame metadata.

        Returns:
            Metadata dictionary associated with the frame.
        """
        return self.metadata

    def add_metadata(self, new_metadata: dict[str, Any]) -> None:
        """Append custom metadata to the frame.

        Args:
            new_metadata: Metadata entries to merge into the frame metadata.
        """
        self.metadata.update(new_metadata)
        
    def update_metadata(self, new_metadata: dict[str, Any]) -> None:
        """Alias for `add_metadata()`.

        Args:
            new_metadata: Metadata entries to merge into the frame metadata.
        """
        self.add_metadata(new_metadata)
        
    @utils._copy_docstring(plots.plot_frame)
    def plot(self, *args: Any, **kwargs: Any) -> Any:
        return plots.plot_frame(self, *args, **kwargs)
        
    @utils._copy_docstring(slice.get_slice)
    def get_slice(self, *args: Any, **kwargs: Any) -> Any:
        return slice.get_slice(self, *args, **kwargs)

    def integrate(self, *args: Any, **kwargs: Any) -> Any:
        """Integrate frame data over time or frequency.

        Args:
            *args: Positional arguments forwarded to `setigen.integrate()`.
            **kwargs: Keyword arguments forwarded to `setigen.integrate()`.

        Returns:
            Integrated array, `Spectrum`, or `TimeSeries`.
        """
        from .integrate import integrate
        return integrate(self, *args, **kwargs)
        
    def get_waterfall(self) -> Any:
        """Return the current frame as an updated waterfall object.

        Returns:
            Waterfall representation of the frame.
        """
        _update_waterfall(self)
        return self.waterfall
    
    def check_waterfall(self) -> Any:
        """Return the updated attached waterfall when one exists.

        Returns:
            Updated waterfall object or `None` when the frame has no attached
            waterfall.
        """
        if self.waterfall is None:
            return None
        return self.get_waterfall()

    def save_fil(self, filename: PathLike, max_load: int = 1) -> None:
        """Save frame data as a SIGPROC filterbank file.

        Args:
            filename: Output `.fil` path.
            max_load: Maximum load parameter for a lazily created waterfall.
        """
        _update_waterfall(self, filename=filename, max_load=max_load)
        _encode_bytestrings(self)
        self.waterfall.write_to_fil(filename)
        _decode_bytestrings(self)

    def save_hdf5(self, filename: PathLike, max_load: int = 1) -> None:
        """Save frame data as an HDF5 waterfall file.

        Args:
            filename: Output `.h5` path.
            max_load: Maximum load parameter for a lazily created waterfall.
        """
        _update_waterfall(self, filename=filename, max_load=max_load)
        _encode_bytestrings(self)
        self.waterfall.write_to_hdf5(filename)
        _decode_bytestrings(self)

    def save_h5(self, filename: PathLike, max_load: int = 1) -> None:
        """Save frame data as an HDF5 waterfall file.

        Args:
            filename: Output `.h5` path.
            max_load: Maximum load parameter for a lazily created waterfall.
        """
        self.save_hdf5(filename, max_load=max_load)

    def save_npy(self, filename: PathLike) -> None:
        """Save frame data as a NumPy binary file.

        Args:
            filename: Output `.npy` path.
        """
        np.save(filename, self.data)

    def load_npy(self, filename: PathLike) -> None:
        """Load frame data from a NumPy binary file.

        Args:
            filename: Input `.npy` path.
        """
        self.data = np.load(filename)

    def save_pickle(self, filename: PathLike) -> None:
        """Serialize the full frame with pickle.

        Args:
            filename: Output pickle path.
        """
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_pickle(cls, filename: PathLike) -> "Frame":
        """Load a frame from a pickled file.

        Args:
            filename: Input pickle path created by `save_pickle()`.

        Returns:
            Deserialized frame object.
        """
        with open(filename, "rb") as f:
            return pickle.load(f)

    
def params_from_backend(obs_length: float = 300, 
                        sample_rate: float = 3e9, 
                        num_branches: int = 1024,
                        fftlength: int = 1048576,
                        int_factor: int = 51) -> dict[str, float]:
    """Return frame parameters implied by backend characteristics.

    Args:
        obs_length: Observation length in seconds.
        sample_rate: Real-voltage sample rate in Hz.
        num_branches: Number of PFB branches.
        fftlength: Fine-channel FFT length.
        int_factor: Fine-channel integration factor.

    Returns:
        Dictionary containing `tchans`, `df`, and `dt`.
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
