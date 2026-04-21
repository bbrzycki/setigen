from __future__ import annotations

import numpy as np

from ._typing import SeedLike

fwhm_m = 2 * np.sqrt(2 * np.log(2))

def fwhm(sigma: float) -> float:
    """Return the Gaussian full width at half maximum.

    Args:
        sigma: Gaussian standard deviation.

    Returns:
        Full width at half maximum.
    """
    return fwhm_m * sigma
    

def gaussian(
    x_mean: float,
    x_std: float,
    shape: int | tuple[int, ...],
    seed: SeedLike = None,
) -> np.ndarray:
    """Sample from a Gaussian distribution.

    Args:
        x_mean: Mean of the distribution.
        x_std: Standard deviation of the distribution.
        shape: Output shape.
        seed: Random seed or generator.

    Returns:
        Sampled Gaussian array.
    """
    rng = np.random.default_rng(seed)
    return rng.normal(x_mean, x_std, shape)


def truncated_gaussian(
    x_mean: float,
    x_std: float,
    x_min: float,
    shape: int | tuple[int, ...],
    seed: SeedLike = None,
) -> np.ndarray:
    """Sample from a Gaussian distribution with a lower bound.

    Args:
        x_mean: Mean of the distribution.
        x_std: Standard deviation of the distribution.
        x_min: Lower bound for returned values.
        shape: Output shape.
        seed: Random seed or generator.

    Returns:
        Lower-bounded Gaussian array.
    """
    return np.maximum(gaussian(x_mean, x_std, shape, seed), x_min)


def chi2(
    x_mean: float,
    chi2_df: int,
    shape: int | tuple[int, ...],
    seed: SeedLike = None,
) -> np.ndarray:
    """Sample a chi-squared distribution scaled to a target mean.

    Args:
        x_mean: Target mean of the scaled distribution.
        chi2_df: Degrees of freedom.
        shape: Output shape.
        seed: Random seed or generator.

    Returns:
        Chi-squared noise array.
    """
    rng = np.random.default_rng(seed)
    return rng.chisquare(df=chi2_df, size=shape) * x_mean / chi2_df
