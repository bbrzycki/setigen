from __future__ import annotations

from abc import ABC, abstractmethod
import pathlib
import shutil
from typing import Any


class SpectrogramBackend(ABC):
    """Private protocol for region-based spectrogram file access."""

    path: pathlib.Path
    mode: str
    header: dict[str, Any]
    shape: tuple[int, int]
    fchans: int
    tchans: int
    df: float
    dt: float
    fch1: float
    ascending: bool
    source_name: str
    t_start: float
    dtype: Any

    @property
    def writable(self) -> bool:
        """Return whether this backend supports writes."""
        return any(flag in self.mode for flag in ("+", "w", "a"))

    @abstractmethod
    def read_region(self,
                    t_start: int,
                    t_stop: int,
                    f_start: int,
                    f_stop: int) -> Any:
        """Read a time/frequency region in internal `Frame` orientation.

        Args:
            t_start: Inclusive time-bin start index.
            t_stop: Exclusive time-bin stop index.
            f_start: Inclusive frequency-channel start index.
            f_stop: Exclusive frequency-channel stop index.

        Returns:
            Two-dimensional array in `Frame` frequency orientation.
        """

    @abstractmethod
    def write_region(self,
                     t_start: int,
                     f_start: int,
                     data: Any) -> None:
        """Write a time/frequency region in internal `Frame` orientation.

        Args:
            t_start: Inclusive time-bin start index.
            f_start: Inclusive frequency-channel start index.
            data: Two-dimensional data block in `Frame` orientation.
        """

    @abstractmethod
    def flush(self) -> None:
        """Flush pending writes to the backing store."""

    @abstractmethod
    def close(self) -> None:
        """Close any open file resources."""

    def __enter__(self) -> "SpectrogramBackend":
        """Return this backend for context-manager use.

        Returns:
            Open backend instance.
        """
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        """Close this backend after context-manager use.

        Args:
            exc_type: Exception type, if the context is exiting with an error.
            exc: Exception instance, if any.
            tb: Traceback, if any.
        """
        self.close()


def _normalize_path(path: str | pathlib.Path) -> pathlib.Path:
    """Normalize a filesystem path for backend access.

    Args:
        path: Input path.

    Returns:
        Expanded absolute path.
    """
    return pathlib.Path(path).expanduser().resolve()


def open_spectrogram(path: str | pathlib.Path, mode: str = "r") -> SpectrogramBackend:
    """Open a spectrogram backend for a supported file type.

    Args:
        path: Input spectrogram path.
        mode: Backend file mode.

    Returns:
        Open spectrogram backend.
    """
    normalized = _normalize_path(path)
    suffix = normalized.suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        from .h5 import H5SpectrogramBackend

        return H5SpectrogramBackend(normalized, mode=mode)
    if suffix == ".fil":
        from .fil import FilSpectrogramBackend

        return FilSpectrogramBackend(normalized, mode=mode)
    raise ValueError(f"Unsupported spectrogram file type: {normalized.suffix}")


def copy_spectrogram(input_path: str | pathlib.Path,
                     output_path: str | pathlib.Path,
                     *,
                     overwrite: bool = False) -> pathlib.Path:
    """Copy a spectrogram file without loading its data into memory.

    Args:
        input_path: Source spectrogram path.
        output_path: Destination spectrogram path.
        overwrite: Whether to replace an existing destination.

    Returns:
        Absolute destination path.
    """
    source = _normalize_path(input_path)
    destination = pathlib.Path(output_path).expanduser().resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination
