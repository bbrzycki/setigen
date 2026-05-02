from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from astropy.stats import sigma_clip


@dataclass(frozen=True)
class NoiseEstimationConfig:
    """Configuration for estimating spectrogram noise statistics."""

    method: str = "sigma_clip"
    sigma: float = 3
    maxiters: int = 5
    context_width: int | float = 2048
    guard_width: int | float = 64
    width_unit: str = "channels"
    combine: str = "pooled"

    def __post_init__(self) -> None:
        """Validate configuration values after dataclass initialization."""
        if self.method not in {"sigma_clip", "median_mad"}:
            raise ValueError("method must be 'sigma_clip' or 'median_mad'")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")
        if self.maxiters < 1:
            raise ValueError("maxiters must be at least 1")
        if self.context_width < 0:
            raise ValueError("context_width must be non-negative")
        if self.guard_width < 0:
            raise ValueError("guard_width must be non-negative")
        if self.width_unit not in {"channels", "channel", "Hz", "hz"}:
            raise ValueError("width_unit must be 'channels' or 'Hz'")
        if self.combine != "pooled":
            raise ValueError("combine currently supports only 'pooled'")


@dataclass(frozen=True)
class NoiseStats:
    """Estimated noise statistics and their selection metadata."""

    mean: float
    std: float
    n_samples: int
    method: str
    sigma: float | None = None
    maxiters: int | None = None
    context_bounds: tuple[int, int] | None = None
    excluded_bounds: tuple[int, int] | None = None
    time_bounds: tuple[int, int] | None = None
    sampled: bool = False

    @property
    def tchans(self) -> int | None:
        """Return the number of time bins represented by these stats.

        Returns:
            Number of time bins when `time_bounds` are known.
        """
        if self.time_bounds is None:
            return None
        return self.time_bounds[1] - self.time_bounds[0]

    def as_tuple(self) -> tuple[float, float]:
        """Return `(mean, std)` for compatibility with legacy callers.

        Returns:
            Mean and standard deviation tuple.
        """
        return self.mean, self.std


def estimate_array_noise_stats(
    data: Any,
    *,
    config: NoiseEstimationConfig | None = None,
    context_bounds: tuple[int, int] | None = None,
    excluded_bounds: tuple[int, int] | None = None,
    time_bounds: tuple[int, int] | None = None,
    sampled: bool = False,
) -> NoiseStats:
    """Estimate noise statistics from an in-memory array.

    Args:
        data: Input intensity samples.
        config: Estimation configuration. Defaults to sigma clipping that
            matches existing eager `Frame` behavior.
        context_bounds: Optional frequency context bounds used to select data.
        excluded_bounds: Optional excluded signal/guard bounds.
        time_bounds: Optional time bounds used to select data.
        sampled: Whether the data are a sample rather than exhaustive selection.

    Returns:
        Structured noise statistics.
    """
    resolved = NoiseEstimationConfig() if config is None else config
    array = np.asarray(data)
    if array.size == 0:
        raise ValueError("Cannot estimate noise statistics from an empty array")

    if resolved.method == "sigma_clip":
        clipped = sigma_clip(array,
                             sigma=resolved.sigma,
                             maxiters=resolved.maxiters,
                             masked=False)
        samples = np.asarray(clipped)
        return NoiseStats(mean=float(np.mean(samples)),
                          std=float(np.std(samples)),
                          n_samples=int(samples.size),
                          method=resolved.method,
                          sigma=resolved.sigma,
                          maxiters=resolved.maxiters,
                          context_bounds=context_bounds,
                          excluded_bounds=excluded_bounds,
                          time_bounds=time_bounds,
                          sampled=sampled)

    flattened = array.ravel()
    median = float(np.median(flattened))
    mad = float(np.median(np.abs(flattened - median)))
    return NoiseStats(mean=median,
                      std=1.4826 * mad,
                      n_samples=int(flattened.size),
                      method=resolved.method,
                      context_bounds=context_bounds,
                      excluded_bounds=excluded_bounds,
                      time_bounds=time_bounds,
                      sampled=sampled)
