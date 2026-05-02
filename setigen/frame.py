from __future__ import annotations

import copy
import pickle
import warnings
from typing import Any

import numpy as np

from astropy import units as u
from astropy.time import Time

from . import unit_utils
from . import slice
from . import plots
from . import utils
from .noise import NoiseEstimationConfig, NoiseStats, estimate_array_noise_stats
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
    _check_waterfall,
    _get_waterfall,
    _save_fil,
    _save_hdf5,
)
from ._frame.file_mutation import (
    FileBackedSignalResult,
    add_signal_to_file_backed_frame,
)
from ._frame.context import _finalize_derived_frame, _source_bounds_metadata
from ._frame.signal import (
    _evaluate_path_values,
    _finalize_signal,
    _get_restricted_fs,
    _normalize_bp_profile,
    _normalize_path,
    _normalize_t_profile,
    _render_signal,
    _resolve_auto_bounding_range,
    _resolve_bounding_indices,
)
from ._spectrogram import copy_spectrogram, open_spectrogram

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
    def open(
        cls,
        path: PathLike,
        *,
        mode: str = "r",
        allow_inplace: bool = False,
        seed: SeedLike = None,
        max_chunk_bytes: int = 256 * 1024 * 1024,
    ) -> "Frame":
        """Open a spectrogram as a file-backed frame.

        Args:
            path: Input `.fil`, `.h5`, or `.hdf5` path.
            mode: File mode. Use `"r"` for read-only access. Writable modes
                require `allow_inplace=True`.
            allow_inplace: Explicit guard for direct mutation of the supplied
                path.
            seed: Random seed or generator.
            max_chunk_bytes: Default memory budget for chunked file-backed
                signal injection.

        Returns:
            File-backed frame.

        Raises:
            ValueError: If a writable mode is requested without
                `allow_inplace=True`.
        """
        if mode not in {"r", "r+"}:
            raise ValueError("Frame.open() currently supports only mode='r' and mode='r+'")
        wants_write = any(flag in mode for flag in ("+", "w", "a"))
        if wants_write and not allow_inplace:
            raise ValueError(
                "Writable file-backed frames can modify their backing file. "
                "Use Frame.open_copy(...) for safe copy-backed mutation, or "
                "pass allow_inplace=True when direct mutation is intended."
            )
        backend = open_spectrogram(path, mode=mode)
        return cls._from_file_backend(backend,
                                      seed=seed,
                                      max_chunk_bytes=max_chunk_bytes)

    @classmethod
    def open_copy(
        cls,
        input_path: PathLike,
        output_path: PathLike,
        *,
        overwrite: bool = False,
        seed: SeedLike = None,
        max_chunk_bytes: int = 256 * 1024 * 1024,
    ) -> "Frame":
        """Create and open a writable copy-backed spectrogram frame.

        The input file is copied on disk without loading the full observation
        into memory. Mutating methods patch the output file immediately.

        Args:
            input_path: Source `.fil`, `.h5`, or `.hdf5` path.
            output_path: Writable output path to create.
            overwrite: Whether to replace an existing output file.
            seed: Random seed or generator.
            max_chunk_bytes: Default memory budget for chunked file-backed
                signal injection.

        Returns:
            File-backed frame whose backing store is `output_path`.
        """
        copied_path = copy_spectrogram(input_path, output_path, overwrite=overwrite)
        backend = open_spectrogram(copied_path, mode="r+")
        return cls._from_file_backend(backend,
                                      seed=seed,
                                      max_chunk_bytes=max_chunk_bytes)

    @classmethod
    def _from_file_backend(
        cls,
        backend: Any,
        *,
        seed: SeedLike = None,
        max_chunk_bytes: int = 256 * 1024 * 1024,
    ) -> "Frame":
        """Build a `Frame` around an already-open file backend.

        Args:
            backend: Open spectrogram backend.
            seed: Random seed or generator.
            max_chunk_bytes: Default memory budget for chunked injection.

        Returns:
            File-backed frame instance.
        """
        frame = cls.__new__(cls)
        frame.rng = np.random.default_rng(seed)
        frame._file_backend = backend
        frame._max_chunk_bytes = max_chunk_bytes
        frame._data = None
        frame.df = backend.df
        frame.dt = backend.dt
        frame.fch1 = backend.fch1
        frame.ascending = backend.ascending
        frame.t_start = backend.t_start
        frame.source_name = backend.source_name
        frame.shape = backend.shape
        frame.tchans, frame.fchans = backend.shape
        frame.waterfall = None
        frame.header = copy.deepcopy(backend.header)
        frame.chi2_df = 4 * round(frame.df * frame.dt)
        frame.unit_drift_rate = frame.df / frame.dt
        frame._update_fs()
        frame._update_ts()
        frame.noise_mean = 0
        frame.noise_std = 0
        frame.noise_stats = None
        frame.metadata = frame.get_params()
        frame.metadata["file_backed"] = True
        frame.metadata["path"] = str(backend.path)
        return frame
    
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
            
        param_dict = frame_params_from_backend(obs_length=obs_length,
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
        if self.is_file_backed:
            return self.read_frame()
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
        state['_file_backend'] = None
        return state

    @property
    def data(self) -> np.ndarray:
        """Return frame data, reading a full file-backed frame when needed."""
        backend = getattr(self, "_file_backend", None)
        if backend is not None:
            return backend.read_region(0, self.tchans, 0, self.fchans)
        return self._data

    @data.setter
    def data(self, value: np.ndarray | None) -> None:
        """Replace frame data, writing through when file-backed.

        Args:
            value: New two-dimensional frame data, or `None`.
        """
        backend = getattr(self, "_file_backend", None)
        if backend is None:
            self._data = value
            return
        if value is None:
            self._data = None
            return
        array = np.asarray(value)
        if array.shape != self.shape:
            raise ValueError(f"Data shape {array.shape} does not match frame shape {self.shape}.")
        backend.write_region(0, 0, array)
        backend.flush()
        self._data = None

    @property
    def is_file_backed(self) -> bool:
        """Whether this frame reads from a backing spectrogram file."""
        return getattr(self, "_file_backend", None) is not None

    def close(self) -> None:
        """Close any file-backed resources owned by this frame."""
        backend = getattr(self, "_file_backend", None)
        if backend is not None:
            backend.close()
            self._file_backend = None

    def __enter__(self) -> "Frame":
        """Return this frame for context-manager use.

        Returns:
            Open frame instance.
        """
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        """Close file-backed resources after context-manager use.

        Args:
            exc_type: Exception type, if any.
            exc: Exception instance, if any.
            tb: Traceback, if any.
        """
        self.close()

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
        if self.is_file_backed:
            self.noise_mean = 0
            self.noise_std = 0
            self.noise_stats = None
            return
        stats = estimate_array_noise_stats(
            self.data,
            time_bounds=(0, self.tchans),
            context_bounds=(0, self.fchans),
        )
        self.noise_stats = stats
        self.noise_mean = stats.mean
        self.noise_std = stats.std

    def _width_to_channels(self,
                           width: int | float,
                           *,
                           width_unit: str = "channels") -> int:
        """Convert a frequency/context width into channel units.

        Args:
            width: Width in channels or frequency units.
            width_unit: Whether `width` is in channels or Hz.

        Returns:
            Non-negative integer channel width.
        """
        if width_unit in {"Hz", "hz"}:
            return max(0, int(np.ceil(unit_utils.get_value(width, u.Hz) / self.df)))
        return max(0, int(np.ceil(width)))

    def _resolve_noise_signal_bounds(
        self,
        *,
        path: FrequencyPathInput | None = None,
        f_profile: FrequencyProfile | None = None,
        bounding_f_range: tuple[Any, Any] | None = None,
        auto_bounding: bool = False,
        truncate_below: float | None = None,
        integrate_path: bool = False,
        integrate_f_profile: bool = False,
        doppler_smearing: bool = False,
        t_subsamples: int = 10,
        t_offset: float = 0,
    ) -> tuple[int, int] | None:
        """Resolve optional signal inputs into frequency-channel bounds.

        Args:
            path: Optional signal path.
            f_profile: Optional frequency profile for automatic bounding.
            bounding_f_range: Optional explicit signal bounding range.
            auto_bounding: Whether to infer bounds for known frequency profiles.
            truncate_below: Optional cutoff for infinite-support profiles.
            integrate_path: Whether path integration is enabled.
            integrate_f_profile: Whether frequency-profile integration is enabled.
            doppler_smearing: Whether Doppler smearing is enabled.
            t_subsamples: Number of time subsamples.
            t_offset: Time offset for callable path evaluation.

        Returns:
            Optional half-open frequency-channel bounds.
        """
        if bounding_f_range is None and auto_bounding:
            if path is None or f_profile is None:
                raise ValueError("auto_bounding noise estimation requires path and f_profile")
            bounding_f_range, _ = _resolve_auto_bounding_range(
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

        if bounding_f_range is not None:
            bounds = _resolve_bounding_indices(self, bounding_f_range)
            if bounds[0] >= bounds[1]:
                center = min(max(bounds[0], 0), self.fchans - 1)
                return center, center + 1
            return bounds

        if path is None:
            return None

        path_values = _evaluate_path_values(self,
                                            path,
                                            integrate_path=integrate_path,
                                            doppler_smearing=doppler_smearing,
                                            t_subsamples=t_subsamples,
                                            t_offset=t_offset)
        bounds = _resolve_bounding_indices(
            self,
            (float(np.min(path_values)), float(np.max(path_values))),
        )
        if bounds[0] >= bounds[1]:
            center = min(max(bounds[0], 0), self.fchans - 1)
            return center, center + 1
        return bounds

    def estimate_noise_stats(
        self,
        *,
        path: FrequencyPathInput | None = None,
        f_profile: FrequencyProfile | None = None,
        bounding_f_range: tuple[Any, Any] | None = None,
        f_range: tuple[Any, Any] | None = None,
        t_range: tuple[Any, Any] | None = None,
        f_index_range: tuple[int, int] | None = None,
        t_index_range: tuple[int, int] | None = None,
        auto_bounding: bool = False,
        truncate_below: float | None = None,
        integrate_path: bool = False,
        integrate_f_profile: bool = False,
        doppler_smearing: bool = False,
        t_subsamples: int = 10,
        t_offset: float = 0,
        config: NoiseEstimationConfig | None = None,
    ) -> NoiseStats:
        """Estimate noise statistics for an eager or file-backed frame.

        Args:
            path: Optional signal path used to define local context.
            f_profile: Optional frequency profile used for automatic bounds.
            bounding_f_range: Optional explicit signal bounding range.
            f_range: Optional direct frequency range for stats.
            t_range: Optional time range for stats.
            f_index_range: Optional direct half-open frequency index range.
            t_index_range: Optional direct half-open time index range.
            auto_bounding: Whether to infer signal bounds for known profiles.
            truncate_below: Optional cutoff for infinite-support profiles.
            integrate_path: Whether path integration is enabled.
            integrate_f_profile: Whether frequency-profile integration is enabled.
            doppler_smearing: Whether Doppler smearing is enabled.
            t_subsamples: Number of time subsamples.
            t_offset: Time offset for callable path evaluation.
            config: Noise-estimation configuration.

        Returns:
            Structured noise statistics.
        """
        resolved_config = NoiseEstimationConfig() if config is None else config
        if self.is_file_backed and all(value is None for value in (
            path,
            bounding_f_range,
            f_range,
            f_index_range,
            t_range,
            t_index_range,
        )):
            raise ValueError(
                "File-backed noise estimation requires an explicit path, "
                "frequency range, or time/frequency index range."
            )

        t_start, t_stop = self._resolve_time_index_range(
            t_range=t_range,
            t_index_range=t_index_range,
        )
        signal_bounds = self._resolve_noise_signal_bounds(
            path=path,
            f_profile=f_profile,
            bounding_f_range=bounding_f_range,
            auto_bounding=auto_bounding,
            truncate_below=truncate_below,
            integrate_path=integrate_path,
            integrate_f_profile=integrate_f_profile,
            doppler_smearing=doppler_smearing,
            t_subsamples=t_subsamples,
            t_offset=t_offset,
        )

        excluded_bounds = None
        if signal_bounds is not None:
            signal_start, signal_stop = signal_bounds
            context_chans = self._width_to_channels(
                resolved_config.context_width,
                width_unit=resolved_config.width_unit,
            )
            guard_chans = self._width_to_channels(
                resolved_config.guard_width,
                width_unit=resolved_config.width_unit,
            )
            f_start = max(0, signal_start - context_chans)
            f_stop = min(self.fchans, signal_stop + context_chans)
            excluded_bounds = (max(0, signal_start - guard_chans),
                               min(self.fchans, signal_stop + guard_chans))
        else:
            f_start, f_stop = self._resolve_frequency_index_range(
                f_range=f_range,
                f_index_range=f_index_range,
            )

        if f_start >= f_stop or t_start >= t_stop:
            raise ValueError("Requested noise-estimation region is empty")

        backend = getattr(self, "_file_backend", None)
        if backend is not None:
            data = backend.read_region(t_start, t_stop, f_start, f_stop)
        else:
            data = np.asarray(self.data[t_start:t_stop, f_start:f_stop])

        if excluded_bounds is not None:
            exclude_start = max(excluded_bounds[0], f_start) - f_start
            exclude_stop = min(excluded_bounds[1], f_stop) - f_start
            mask = np.ones(data.shape, dtype=bool)
            mask[:, exclude_start:exclude_stop] = False
            data = data[mask]
            if data.size == 0:
                raise ValueError("Noise-estimation guard removed all context samples")

        return estimate_array_noise_stats(data,
                                          config=resolved_config,
                                          context_bounds=(f_start, f_stop),
                                          excluded_bounds=excluded_bounds,
                                          time_bounds=(t_start, t_stop))

    def zero_data(self) -> None:
        """Reset frame data and cached noise statistics to zero."""
        self.data = np.zeros(self.shape)
        self.noise_mean = self.noise_std = 0
        self.noise_stats = NoiseStats(mean=0,
                                      std=0,
                                      n_samples=int(np.prod(self.shape)),
                                      method="constant",
                                      context_bounds=(0, self.fchans),
                                      time_bounds=(0, self.tchans))

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
        if self.is_file_backed:
            raise NotImplementedError(
                "add_noise() is not implemented for file-backed frames because "
                "it would require full-observation mutation. Read a region with "
                "read_frame() or use an eager Frame for synthetic noise."
            )
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
            self.noise_stats = NoiseStats(mean=float(x_mean),
                                          std=float(x_std),
                                          n_samples=int(np.prod(self.shape)),
                                          method="parameter",
                                          context_bounds=(0, self.fchans),
                                          time_bounds=(0, self.tchans))
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
        if self.is_file_backed:
            raise NotImplementedError(
                "add_noise_from_obs() is not implemented for file-backed frames. "
                "Use read_frame() for a bounded eager region first."
            )
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
            self.noise_stats = NoiseStats(mean=float(x_mean),
                                          std=float(x_std),
                                          n_samples=int(np.prod(self.shape)),
                                          method="sampled_parameter",
                                          context_bounds=(0, self.fchans),
                                          time_bounds=(0, self.tchans))
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
                   truncate_below: float | None = None,
                   max_chunk_bytes: int | None = None,
                   chunk_tchans: int | None = None) -> np.ndarray | FileBackedSignalResult:
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
            max_chunk_bytes: Optional memory budget for file-backed injection.
            chunk_tchans: Optional time-chunk size for file-backed injection.

        Returns:
            Two-dimensional signal array that was added to an in-memory frame,
            or a file-backed injection summary for file-backed frames.
        """
        if self.is_file_backed:
            return add_signal_to_file_backed_frame(
                self,
                path=path,
                t_profile=t_profile,
                f_profile=f_profile,
                bp_profile=bp_profile,
                bounding_f_range=bounding_f_range,
                integrate_path=integrate_path,
                integrate_t_profile=integrate_t_profile,
                integrate_f_profile=integrate_f_profile,
                doppler_smearing=doppler_smearing,
                t_subsamples=t_subsamples,
                f_subsamples=f_subsamples,
                smearing_subsamples=smearing_subsamples,
                t_offset=t_offset,
                auto_bounding=auto_bounding,
                truncate_below=truncate_below,
                max_chunk_bytes=max_chunk_bytes,
                chunk_tchans=chunk_tchans,
            )

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
                            doppler_smearing: bool = False) -> np.ndarray | FileBackedSignalResult:
        """Add a constant-intensity, constant-drift signal to the frame.

        Args:
            f_start: Starting signal frequency.
            drift_rate: Signal drift rate.
            level: Signal intensity.
            width: Signal width.
            f_profile_type: Spectral profile selector.
            doppler_smearing: Whether to numerically smear power across bins.

        Returns:
            Two-dimensional signal array for in-memory frames, or a
            file-backed injection summary for file-backed frames.
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

    def get_intensity(self,
                      snr: float,
                      noise_stats: NoiseStats | tuple[float, float] | None = None) -> float:
        """Calculate signal intensity from SNR using the frame noise estimate.

        Args:
            snr: Desired signal-to-noise ratio.
            noise_stats: Optional explicit noise statistics.

        Returns:
            Signal intensity that corresponds to the requested SNR.

        Raises:
            ValueError: If the frame does not yet contain measurable noise.
        """
        if noise_stats is None:
            noise_std = self.noise_std
            tchans = self.tchans
        elif isinstance(noise_stats, NoiseStats):
            noise_std = noise_stats.std
            tchans = noise_stats.tchans or self.tchans
        else:
            noise_std = noise_stats[1]
            tchans = self.tchans

        if noise_std == 0:
            raise ValueError('You must add noise in the image to specify SNR!')
        return snr * noise_std / np.sqrt(tchans)

    def get_snr(self,
                intensity: float,
                noise_stats: NoiseStats | tuple[float, float] | None = None) -> float:
        """Calculate SNR from signal intensity using the frame noise estimate.

        Args:
            intensity: Signal intensity.
            noise_stats: Optional explicit noise statistics.

        Returns:
            Signal-to-noise ratio for the supplied intensity.

        Raises:
            ValueError: If the frame does not yet contain measurable noise.
        """
        if noise_stats is None:
            noise_std = self.noise_std
            tchans = self.tchans
        elif isinstance(noise_stats, NoiseStats):
            noise_std = noise_stats.std
            tchans = noise_stats.tchans or self.tchans
        else:
            noise_std = noise_stats[1]
            tchans = self.tchans

        if noise_std == 0:
            raise ValueError('You must add noise in the image to return SNR!')
        return intensity * np.sqrt(tchans) / noise_std

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
        data = self.data
        if db:
            return 10 * np.log10(data)
        return data

    def _resolve_frequency_index_range(
        self,
        *,
        f_range: tuple[Any, Any] | None = None,
        f_index_range: tuple[int, int] | None = None,
    ) -> tuple[int, int]:
        """Resolve frequency selection inputs to half-open channel bounds.

        Args:
            f_range: Optional frequency range in Hz or frequency units.
            f_index_range: Optional half-open frequency index range.

        Returns:
            Clipped half-open frequency index bounds.
        """
        if f_index_range is not None:
            start, stop = f_index_range
            return max(0, int(start)), min(self.fchans, int(stop))
        if f_range is None:
            return 0, self.fchans
        f0 = unit_utils.get_value(f_range[0], u.Hz)
        f1 = unit_utils.get_value(f_range[1], u.Hz)
        f_min, f_max = sorted((f0, f1))
        start = int(np.searchsorted(self.fs, f_min, side="left"))
        stop = int(np.searchsorted(self.fs, f_max, side="right"))
        return max(0, start), min(self.fchans, stop)

    def _resolve_time_index_range(
        self,
        *,
        t_range: tuple[Any, Any] | None = None,
        t_index_range: tuple[int, int] | None = None,
    ) -> tuple[int, int]:
        """Resolve time selection inputs to half-open time-bin bounds.

        Args:
            t_range: Optional time range in seconds or time units.
            t_index_range: Optional half-open time index range.

        Returns:
            Clipped half-open time index bounds.
        """
        if t_index_range is not None:
            start, stop = t_index_range
            return max(0, int(start)), min(self.tchans, int(stop))
        if t_range is None:
            return 0, self.tchans
        t0 = unit_utils.get_value(t_range[0], u.s)
        t1 = unit_utils.get_value(t_range[1], u.s)
        t_min, t_max = sorted((t0, t1))
        start = int(np.floor(t_min / self.dt))
        stop = int(np.ceil(t_max / self.dt))
        return max(0, start), min(self.tchans, stop)

    def read_frame(
        self,
        *,
        f_range: tuple[Any, Any] | None = None,
        t_range: tuple[Any, Any] | None = None,
        f_index_range: tuple[int, int] | None = None,
        t_index_range: tuple[int, int] | None = None,
    ) -> "Frame":
        """Read a time/frequency region as an eager in-memory frame.

        Args:
            f_range: Optional frequency range in Hz or frequency units.
            t_range: Optional time range in seconds or time units, relative to
                this frame start.
            f_index_range: Optional half-open frequency index range.
            t_index_range: Optional half-open time index range.

        Returns:
            Eager `Frame` containing the requested region.
        """
        f_start, f_stop = self._resolve_frequency_index_range(
            f_range=f_range,
            f_index_range=f_index_range,
        )
        t_start, t_stop = self._resolve_time_index_range(
            t_range=t_range,
            t_index_range=t_index_range,
        )
        if f_start >= f_stop or t_start >= t_stop:
            raise ValueError("Requested frame region is empty")

        backend = getattr(self, "_file_backend", None)
        if backend is not None:
            data = backend.read_region(t_start, t_stop, f_start, f_stop)
        else:
            data = np.array(self.data[t_start:t_stop, f_start:f_stop], copy=True)

        if self.ascending:
            fch1 = self.fs[f_start]
        else:
            fch1 = self.fs[f_stop - 1]

        new_frame = Frame.from_data(
            df=self.df,
            dt=self.dt,
            fch1=fch1,
            ascending=self.ascending,
            data=data,
            seed=self.rng,
            t_start=self.t_start + t_start * self.dt,
            source_name=self.source_name,
        )
        _finalize_derived_frame(
            self,
            new_frame,
            operation="read_frame",
            product_type="frame",
            source_bounds=_source_bounds_metadata(
                self,
                f_index_range=(f_start, f_stop),
                t_index_range=(t_start, t_stop),
            ),
        )
        return new_frame

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

    def spectrum(self, *args: Any, **kwargs: Any) -> Any:
        """Integrate this frame over time and return a `Spectrum`.

        Args:
            *args: Positional arguments forwarded to `setigen.spectrum()`.
            **kwargs: Keyword arguments forwarded to `setigen.spectrum()`.

        Returns:
            Integrated spectrum.
        """
        from .integrate import spectrum
        return spectrum(self, *args, **kwargs)

    def timeseries(self, *args: Any, **kwargs: Any) -> Any:
        """Integrate this frame over frequency and return a `TimeSeries`.

        Args:
            *args: Positional arguments forwarded to `setigen.timeseries()`.
            **kwargs: Keyword arguments forwarded to `setigen.timeseries()`.

        Returns:
            Integrated time series.
        """
        from .integrate import timeseries
        return timeseries(self, *args, **kwargs)
        
    def get_waterfall(self) -> Any:
        """Return the current frame as an updated waterfall object.

        Returns:
            Waterfall representation of the frame.
        """
        return _get_waterfall(self)
    
    def check_waterfall(self) -> Any:
        """Return the updated attached waterfall when one exists.

        Returns:
            Updated waterfall object or `None` when the frame has no attached
            waterfall.
        """
        return _check_waterfall(self)

    def save_fil(self, filename: PathLike, max_load: int = 1) -> None:
        """Save frame data as a SIGPROC filterbank file.

        Args:
            filename: Output `.fil` path.
            max_load: Maximum load parameter for a lazily created waterfall.
        """
        _save_fil(self, filename, max_load=max_load)

    def save_hdf5(self, filename: PathLike, max_load: int = 1) -> None:
        """Save frame data as an HDF5 waterfall file.

        Args:
            filename: Output `.h5` path.
            max_load: Maximum load parameter for a lazily created waterfall.
        """
        _save_hdf5(self, filename, max_load=max_load)

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

    
def frame_params_from_backend(obs_length: float = 300,
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
        Dictionary containing `tchans`, `df`, and `dt` suitable for expansion
        into `Frame(...)`.
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


def params_from_backend(obs_length: float = 300,
                        sample_rate: float = 3e9,
                        num_branches: int = 1024,
                        fftlength: int = 1048576,
                        int_factor: int = 51) -> dict[str, float]:
    """Return frame parameters implied by backend characteristics.

    Deprecated:
        Use `frame_params_from_backend()` instead.

    Args:
        obs_length: Observation length in seconds.
        sample_rate: Real-voltage sample rate in Hz.
        num_branches: Number of PFB branches.
        fftlength: Fine-channel FFT length.
        int_factor: Fine-channel integration factor.

    Returns:
        Dictionary containing `tchans`, `df`, and `dt` suitable for expansion
        into `Frame(...)`.
    """
    warnings.warn(
        "params_from_backend() is deprecated; use frame_params_from_backend() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return frame_params_from_backend(obs_length=obs_length,
                                     sample_rate=sample_rate,
                                     num_branches=num_branches,
                                     fftlength=fftlength,
                                     int_factor=int_factor)
