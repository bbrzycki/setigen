from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import pathlib
from typing import Any

import numpy as np

from .. import distributions
from .. import sample_from_obs
from ..funcs import bp_profiles
from ..funcs import f_profiles
from ..funcs import paths
from ..funcs import t_profiles


class _NoiseType(str, Enum):
    """Supported synthetic noise distributions."""

    CHI2 = "chi2"
    GAUSSIAN = "gaussian"


def _coerce_noise_type(noise_type: str | _NoiseType) -> _NoiseType:
    """Normalize a user-supplied noise-type value.

    Args:
        noise_type: Raw noise-type value from the caller.

    Returns:
        Normalized noise-type enum value.

    Raises:
        ValueError: If the value is not a supported noise type.
    """
    if isinstance(noise_type, _NoiseType):
        return noise_type
    if noise_type == "normal":
        return _NoiseType.GAUSSIAN
    try:
        return _NoiseType(noise_type)
    except ValueError as exc:
        raise ValueError(f"'{noise_type}' is not a valid noise type") from exc


@dataclass(frozen=True)
class _NoiseConfig:
    """Normalized configuration for synthetic noise generation."""

    x_mean: float
    x_std: float | None = None
    x_min: float | None = None
    noise_type: _NoiseType = _NoiseType.CHI2

    @classmethod
    def from_values(cls,
                    x_mean: float,
                    x_std: float | None = None,
                    x_min: float | None = None,
                    noise_type: str | _NoiseType = "chi2") -> "_NoiseConfig":
        """Build a normalized noise configuration from raw caller values.

        Args:
            x_mean: Target mean.
            x_std: Target standard deviation for Gaussian noise.
            x_min: Lower bound for truncated Gaussian noise.
            noise_type: Noise distribution selector.

        Returns:
            Normalized noise configuration.
        """
        return cls(x_mean=x_mean,
                   x_std=x_std,
                   x_min=x_min,
                   noise_type=_coerce_noise_type(noise_type))


def _generate_noise(
    config: _NoiseConfig,
    *,
    chi2_df: int,
    shape: tuple[int, int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, float]:
    """Generate synthetic frame noise from a normalized configuration.

    Args:
        config: Normalized noise configuration.
        chi2_df: Degrees of freedom for chi-squared radiometer noise.
        shape: Output frame shape.
        rng: Random generator used for sampling.

    Returns:
        Generated noise, mean level, and standard deviation.

    Raises:
        ValueError: If Gaussian noise is requested without a standard deviation.
    """
    if config.noise_type is _NoiseType.CHI2:
        noise = distributions.chi2(config.x_mean,
                                   chi2_df,
                                   shape,
                                   seed=rng)
        x_std = np.sqrt(2 * chi2_df) * config.x_mean / chi2_df
    else:
        if config.x_std is None:
            raise ValueError("x_std must be given")
        if config.x_min is not None:
            noise = distributions.truncated_gaussian(config.x_mean,
                                                     config.x_std,
                                                     config.x_min,
                                                     shape,
                                                     seed=rng)
        else:
            noise = distributions.gaussian(config.x_mean,
                                           config.x_std,
                                           shape,
                                           seed=rng)
        x_std = config.x_std
    return noise, config.x_mean, x_std


@dataclass(frozen=True)
class _SampledNoiseConfig:
    """Normalized configuration for sampling empirical noise parameters."""

    x_mean_array: np.ndarray | None = None
    x_std_array: np.ndarray | None = None
    x_min_array: np.ndarray | None = None
    share_index: bool = True
    noise_type: _NoiseType = _NoiseType.CHI2

    @classmethod
    def from_values(cls,
                    x_mean_array: np.ndarray | None = None,
                    x_std_array: np.ndarray | None = None,
                    x_min_array: np.ndarray | None = None,
                    share_index: bool = True,
                    noise_type: str | _NoiseType = "chi2") -> "_SampledNoiseConfig":
        """Build a normalized sampled-noise configuration from raw values.

        Args:
            x_mean_array: Candidate noise means.
            x_std_array: Candidate noise standard deviations.
            x_min_array: Candidate truncated-Gaussian minima.
            share_index: Whether to sample correlated parameters by shared index.
            noise_type: Noise distribution selector.

        Returns:
            Normalized sampled-noise configuration.
        """
        return cls(x_mean_array=x_mean_array,
                   x_std_array=x_std_array,
                   x_min_array=x_min_array,
                   share_index=share_index,
                   noise_type=_coerce_noise_type(noise_type))


def _resolve_sample_noise_arrays(
    config: _SampledNoiseConfig,
    *,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Resolve empirical noise-parameter arrays for sampled noise generation.

    Args:
        config: Sampled-noise configuration.
        dt: Frame time resolution in seconds.

    Returns:
        Mean, standard-deviation, and minimum arrays scaled for the frame.
    """
    if (config.x_mean_array is None
        and config.x_std_array is None
            and config.x_min_array is None):
        path = pathlib.Path(__file__).resolve().parents[1] / "assets" / "sample_noise_params.npy"
        sample_noise_params = np.load(path)

        obs_dt = 1.4316557653333333
        scale_factor = dt / obs_dt

        return (sample_noise_params[:, 0] * scale_factor,
                sample_noise_params[:, 1] * scale_factor,
                sample_noise_params[:, 2] * scale_factor)

    return config.x_mean_array, config.x_std_array, config.x_min_array


def _generate_sampled_noise(
    config: _SampledNoiseConfig,
    *,
    dt: float,
    chi2_df: int,
    shape: tuple[int, int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, float]:
    """Generate sampled synthetic noise from empirical parameter arrays.

    Args:
        config: Sampled-noise configuration.
        dt: Frame time resolution in seconds.
        chi2_df: Degrees of freedom for chi-squared radiometer noise.
        shape: Output frame shape.
        rng: Random generator used for sampling.

    Returns:
        Generated noise, mean level, and standard deviation.

    Raises:
        IndexError: If shared-index sampling is requested for mismatched arrays.
    """
    x_mean_array, x_std_array, x_min_array = _resolve_sample_noise_arrays(config,
                                                                          dt=dt)

    if config.noise_type is _NoiseType.CHI2:
        x_mean = rng.choice(x_mean_array)
        noise = distributions.chi2(x_mean,
                                   chi2_df,
                                   shape,
                                   seed=rng)
        x_std = np.sqrt(2 * chi2_df) * x_mean / chi2_df
        return noise, x_mean, x_std

    if x_min_array is not None:
        if config.share_index:
            if (len(x_mean_array) != len(x_std_array)
                    or len(x_mean_array) != len(x_min_array)):
                raise IndexError("To share a random index, all parameter arrays must be the same length!")
            i = rng.integers(len(x_mean_array))
            x_mean, x_std, x_min = x_mean_array[i], x_std_array[i], x_min_array[i]
        else:
            x_mean, x_std, x_min = sample_from_obs.sample_gaussian_params(x_mean_array,
                                                                          x_std_array,
                                                                          x_min_array,
                                                                          seed=rng)
        noise = distributions.truncated_gaussian(x_mean,
                                                 x_std,
                                                 x_min,
                                                 shape,
                                                 seed=rng)
        return noise, x_mean, x_std

    if config.share_index:
        if len(x_mean_array) != len(x_std_array):
            raise IndexError("To share a random index, all parameter arrays must be the same length!")
        i = rng.integers(len(x_mean_array))
        x_mean, x_std = x_mean_array[i], x_std_array[i]
    else:
        x_mean, x_std = sample_from_obs.sample_gaussian_params(x_mean_array,
                                                               x_std_array,
                                                               seed=rng)

    noise = distributions.gaussian(x_mean,
                                   x_std,
                                   shape,
                                   seed=rng)
    return noise, x_mean, x_std


class _FrequencyProfileType(str, Enum):
    """Supported constant-signal spectral profile families."""

    GAUSSIAN = "gaussian"
    LORENTZIAN = "lorentzian"
    VOIGT = "voigt"
    SINC2 = "sinc2"
    BOX = "box"


def _coerce_frequency_profile_type(
    f_profile_type: str | _FrequencyProfileType,
) -> _FrequencyProfileType:
    """Normalize a user-supplied constant-signal profile type.

    Args:
        f_profile_type: Raw profile selector from the caller.

    Returns:
        Normalized profile-type enum value.

    Raises:
        ValueError: If the requested profile type is unsupported.
    """
    if isinstance(f_profile_type, _FrequencyProfileType):
        return f_profile_type
    try:
        return _FrequencyProfileType(f_profile_type)
    except ValueError as exc:
        raise ValueError("Unsupported f_profile for constant signal!") from exc


@dataclass(frozen=True)
class _ConstantSignalConfig:
    """Normalized configuration for a constant drifting signal."""

    f_start: float
    drift_rate: float
    level: float
    width: float
    f_profile_type: _FrequencyProfileType = _FrequencyProfileType.SINC2
    doppler_smearing: bool = False

    @classmethod
    def from_values(cls,
                    f_start: float,
                    drift_rate: float,
                    level: float,
                    width: float,
                    f_profile_type: str | _FrequencyProfileType = "sinc2",
                    doppler_smearing: bool = False) -> "_ConstantSignalConfig":
        """Build a normalized constant-signal configuration.

        Args:
            f_start: Starting frequency in Hz.
            drift_rate: Drift rate in Hz/s.
            level: Signal intensity.
            width: Signal width in Hz.
            f_profile_type: Spectral profile selector.
            doppler_smearing: Whether to enable numerical Doppler smearing.

        Returns:
            Normalized constant-signal configuration.
        """
        return cls(f_start=f_start,
                   drift_rate=drift_rate,
                   level=level,
                   width=width,
                   f_profile_type=_coerce_frequency_profile_type(f_profile_type),
                   doppler_smearing=doppler_smearing)


def _resolve_constant_signal_profile(config: _ConstantSignalConfig) -> Any:
    """Resolve the callable frequency profile for a constant signal.

    Args:
        config: Constant-signal configuration.

    Returns:
        Frequency-profile callable matching the configured profile type.
    """
    if config.f_profile_type is _FrequencyProfileType.GAUSSIAN:
        return f_profiles.gaussian_f_profile(config.width)
    if config.f_profile_type is _FrequencyProfileType.LORENTZIAN:
        return f_profiles.lorentzian_f_profile(config.width)
    if config.f_profile_type is _FrequencyProfileType.VOIGT:
        return f_profiles.voigt_f_profile(config.width, config.width)
    if config.f_profile_type is _FrequencyProfileType.SINC2:
        return f_profiles.sinc2_f_profile(config.width)
    return f_profiles.box_f_profile(config.width)


def _build_constant_signal_kwargs(
    frame: Any,
    config: _ConstantSignalConfig,
) -> dict[str, Any]:
    """Build `Frame.add_signal()` kwargs for a constant drifting signal.

    Args:
        frame: Frame instance defining the pixel scale.
        config: Constant-signal configuration.

    Returns:
        Keyword arguments ready for `Frame.add_signal()`.
    """
    start_index = frame.get_index(config.f_start)

    px_width_offset = 2 * config.width / frame.df
    if config.drift_rate < 0:
        px_width_offset = -px_width_offset
    px_drift_offset = frame.dt * (frame.tchans - 1) * config.drift_rate / frame.df
    if config.doppler_smearing:
        px_drift_offset += config.drift_rate * frame.dt / frame.df

    bounding_start_index = start_index + int(-px_width_offset)
    bounding_stop_index = start_index + int(px_drift_offset + px_width_offset)

    bounding_min_index = max(min(bounding_start_index, bounding_stop_index), 0)
    bounding_max_index = min(max(bounding_start_index, bounding_stop_index), frame.fchans)

    return {
        "path": paths.constant_path(config.f_start, config.drift_rate),
        "t_profile": t_profiles.constant_t_profile(config.level),
        "f_profile": _resolve_constant_signal_profile(config),
        "bp_profile": bp_profiles.constant_bp_profile(level=1),
        "bounding_f_range": (frame.get_frequency(bounding_min_index),
                              frame.get_frequency(bounding_max_index)),
        "doppler_smearing": config.doppler_smearing,
        "smearing_subsamples": max(1, int(np.ceil(abs(config.drift_rate / frame.unit_drift_rate)))),
    }
