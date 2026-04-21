from __future__ import annotations

import numpy as np
from astropy.stats import sigma_clip

from . import waterfall_utils
from . import split_utils
from ._typing import PathLike, SeedLike


def sample_gaussian_params(
    x_mean_array: np.ndarray,
    x_std_array: np.ndarray,
    x_min_array: np.ndarray | None = None,
    seed: SeedLike = None,
) -> tuple[float, float] | tuple[float, float, float]:
    """Sample Gaussian parameters from empirical distributions.

    Args:
        x_mean_array: Candidate distribution means.
        x_std_array: Candidate distribution standard deviations.
        x_min_array: Optional candidate lower bounds.
        seed: Random seed or generator.

    Returns:
        Tuple containing sampled mean and standard deviation, plus minimum when
        `x_min_array` is provided.
    """
    rng = np.random.default_rng(seed)
    x_mean = rng.choice(x_mean_array)
    x_std = rng.choice(x_std_array)

    # Somewhat arbitrary decision to ensure that the mean is at least the
    # standard deviation
    x_mean = np.maximum(x_mean, x_std)

    if x_min_array is not None:
        x_min = rng.choice(x_min_array)
        return x_mean, x_std, x_min

    return x_mean, x_std


def get_parameter_distributions(
    waterfall_fn: PathLike,
    fchans: int,
    tchans: int | None = None,
    f_shift: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate empirical noise-parameter distributions from observations.

    Args:
        waterfall_fn: Input filterbank filename.
        fchans: Number of frequency samples per split frame.
        tchans: Optional number of time samples to include.
        f_shift: Optional shift in frequency bins between splits.

    Returns:
        Arrays of empirical means, standard deviations, and minima.
    """
    split_generator = split_utils.split_waterfall_generator(waterfall_fn,
                                                            fchans,
                                                            tchans=tchans,
                                                            f_shift=f_shift)

    x_mean_array = []
    x_std_array = []
    x_min_array = []
    for waterfall in split_generator:
        clipped_data = sigma_clip(waterfall_utils.get_data(waterfall),
                                  sigma=3,
                                  maxiters=5,
                                  masked=False)
        x_mean_array.append(np.mean(clipped_data))
        x_std_array.append(np.std(clipped_data))
        x_min_array.append(np.min(clipped_data))

    x_mean_array = np.array(x_mean_array)
    x_std_array = np.array(x_std_array)
    x_min_array = np.array(x_min_array)

    return (x_mean_array, x_std_array, x_min_array)


def get_mean_distribution(
    waterfall_fn: PathLike,
    fchans: int,
    tchans: int | None = None,
    f_shift: int | None = None,
) -> np.ndarray:
    """Estimate an empirical distribution of mean intensities.

    Args:
        waterfall_fn: Input filterbank filename.
        fchans: Number of frequency samples per split frame.
        tchans: Optional number of time samples to include.
        f_shift: Optional shift in frequency bins between splits.

    Returns:
        Array of empirical mean intensities.
    """
    split_generator = split_utils.split_waterfall_generator(waterfall_fn,
                                                            fchans,
                                                            tchans=tchans,
                                                            f_shift=f_shift)

    x_mean_array = []
    for waterfall in split_generator:
        clipped_data = sigma_clip(waterfall_utils.get_data(waterfall),
                                  sigma=3,
                                  maxiters=5,
                                  masked=False)
        x_mean_array.append(np.mean(clipped_data))

    x_mean_array = np.array(x_mean_array)

    return x_mean_array
