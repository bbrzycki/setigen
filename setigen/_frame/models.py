from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import pathlib

import numpy as np

from .. import distributions
from .. import sample_from_obs
from ..funcs import bp_profiles
from ..funcs import f_profiles
from ..funcs import paths
from ..funcs import t_profiles


class _NoiseType(str, Enum):
    CHI2 = "chi2"
    GAUSSIAN = "gaussian"


def _coerce_noise_type(noise_type):
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
    x_mean: float
    x_std: float | None = None
    x_min: float | None = None
    noise_type: _NoiseType = _NoiseType.CHI2

    @classmethod
    def from_values(cls,
                    x_mean,
                    x_std=None,
                    x_min=None,
                    noise_type="chi2"):
        return cls(x_mean=x_mean,
                   x_std=x_std,
                   x_min=x_min,
                   noise_type=_coerce_noise_type(noise_type))


def _generate_noise(config, *, chi2_df, shape, rng):
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
    x_mean_array: object = None
    x_std_array: object = None
    x_min_array: object = None
    share_index: bool = True
    noise_type: _NoiseType = _NoiseType.CHI2

    @classmethod
    def from_values(cls,
                    x_mean_array=None,
                    x_std_array=None,
                    x_min_array=None,
                    share_index=True,
                    noise_type="chi2"):
        return cls(x_mean_array=x_mean_array,
                   x_std_array=x_std_array,
                   x_min_array=x_min_array,
                   share_index=share_index,
                   noise_type=_coerce_noise_type(noise_type))


def _resolve_sample_noise_arrays(config, *, dt):
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


def _generate_sampled_noise(config, *, dt, chi2_df, shape, rng):
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
    GAUSSIAN = "gaussian"
    LORENTZIAN = "lorentzian"
    VOIGT = "voigt"
    SINC2 = "sinc2"
    BOX = "box"


def _coerce_frequency_profile_type(f_profile_type):
    if isinstance(f_profile_type, _FrequencyProfileType):
        return f_profile_type
    try:
        return _FrequencyProfileType(f_profile_type)
    except ValueError as exc:
        raise ValueError("Unsupported f_profile for constant signal!") from exc


@dataclass(frozen=True)
class _ConstantSignalConfig:
    f_start: float
    drift_rate: float
    level: float
    width: float
    f_profile_type: _FrequencyProfileType = _FrequencyProfileType.SINC2
    doppler_smearing: bool = False

    @classmethod
    def from_values(cls,
                    f_start,
                    drift_rate,
                    level,
                    width,
                    f_profile_type="sinc2",
                    doppler_smearing=False):
        return cls(f_start=f_start,
                   drift_rate=drift_rate,
                   level=level,
                   width=width,
                   f_profile_type=_coerce_frequency_profile_type(f_profile_type),
                   doppler_smearing=doppler_smearing)


def _resolve_constant_signal_profile(config):
    if config.f_profile_type is _FrequencyProfileType.GAUSSIAN:
        return f_profiles.gaussian_f_profile(config.width)
    if config.f_profile_type is _FrequencyProfileType.LORENTZIAN:
        return f_profiles.lorentzian_f_profile(config.width)
    if config.f_profile_type is _FrequencyProfileType.VOIGT:
        return f_profiles.voigt_f_profile(config.width, config.width)
    if config.f_profile_type is _FrequencyProfileType.SINC2:
        return f_profiles.sinc2_f_profile(config.width)
    return f_profiles.box_f_profile(config.width)


def _build_constant_signal_kwargs(frame, config):
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
        "smearing_subsamples": int(np.ceil(config.drift_rate / frame.unit_drift_rate)),
    }
