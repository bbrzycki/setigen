from __future__ import annotations

from os import PathLike as OsPathLike
from typing import Protocol, TypeAlias

import numpy as np
from numpy.random import BitGenerator, Generator, SeedSequence


PathLike: TypeAlias = str | OsPathLike[str]
SeedLike: TypeAlias = None | int | Generator | BitGenerator | SeedSequence
ArrayLike: TypeAlias = np.ndarray
ScalarLike: TypeAlias = int | float | np.number
ScalarOrArray: TypeAlias = ScalarLike | np.ndarray


class FrequencyPath(Protocol):
    """Callable path in time-frequency space."""

    def __call__(self, ts: np.ndarray) -> ScalarOrArray:
        """Evaluate the path at the provided times.

        Args:
            ts: Sample times in seconds.

        Returns:
            Frequency path evaluated at ``ts``.
        """
        ...


class TimeProfile(Protocol):
    """Callable time-intensity profile."""

    def __call__(self, ts: np.ndarray) -> ScalarOrArray:
        """Evaluate the profile at the provided times.

        Args:
            ts: Sample times in seconds.

        Returns:
            Profile values evaluated at ``ts``.
        """
        ...


class FrequencyProfile(Protocol):
    """Callable frequency profile centered on a signal path."""

    def __call__(self, freqs: np.ndarray, center_freqs: np.ndarray) -> np.ndarray:
        """Evaluate the profile for frequencies around a signal path.

        Args:
            freqs: Frequency mesh to evaluate.
            center_freqs: Center frequencies that define the signal path.

        Returns:
            Frequency-profile response values.
        """
        ...


class BandpassProfile(Protocol):
    """Callable bandpass profile over frequency."""

    def __call__(self, freqs: np.ndarray) -> ScalarOrArray:
        """Evaluate the bandpass profile over the provided frequencies.

        Args:
            freqs: Frequencies to evaluate.

        Returns:
            Bandpass response values.
        """
        ...


FrequencyPathInput: TypeAlias = FrequencyPath | np.ndarray | list[float] | float | int
TimeProfileInput: TypeAlias = TimeProfile | np.ndarray | list[float] | float | int
BandpassProfileInput: TypeAlias = BandpassProfile | np.ndarray | list[float] | float | int
