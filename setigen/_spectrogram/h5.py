from __future__ import annotations

import copy
import pathlib
from typing import Any

import numpy as np
from astropy.time import Time

try:  # Register HDF5 filters used by BL-style products when available.
    import hdf5plugin  # noqa: F401
except Exception:  # pragma: no cover - uncompressed HDF5 files do not need it.
    pass

import h5py

from .base import SpectrogramBackend


def _coerce_attr(value: Any) -> Any:
    """Convert HDF5 attribute scalar values into plain Python values.

    Args:
        value: Raw HDF5 attribute value.

    Returns:
        Decoded or scalarized value when possible.
    """
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.generic):
        return value.item()
    return value


class H5SpectrogramBackend(SpectrogramBackend):
    """Region reader/writer for BL-style HDF5 waterfall files."""

    def __init__(self, path: pathlib.Path, mode: str = "r") -> None:
        """Open an HDF5 spectrogram backend.

        Args:
            path: HDF5 file path.
            mode: HDF5 open mode.
        """
        self.path = path
        self.mode = mode
        self._h5 = h5py.File(path, mode)
        self._dataset = self._h5["data"]
        self.header = {
            key: _coerce_attr(value)
            for key, value in self._dataset.attrs.items()
        }
        self._initialize_metadata()

    def _initialize_metadata(self) -> None:
        """Populate frame-compatible metadata from the HDF5 dataset."""
        shape = self._dataset.shape
        if len(shape) == 3:
            self.tchans, self.nifs, self.fchans = shape
        elif len(shape) == 2:
            self.tchans, self.fchans = shape
            self.nifs = int(self.header.get("nifs", 1))
        else:
            raise ValueError(f"Unsupported HDF5 data shape: {shape}")

        self.shape = (int(self.tchans), int(self.fchans))
        self.header.setdefault("nchans", self.fchans)
        self.header.setdefault("nifs", self.nifs)
        self.dtype = self._dataset.dtype
        self.df = abs(float(self.header["foff"])) * 1e6
        self.dt = float(self.header["tsamp"])
        self.ascending = float(self.header["foff"]) > 0
        self.fch1 = float(self.header["fch1"]) * 1e6
        self.t_start = Time(float(self.header["tstart"]), format="mjd").unix
        source_name = self.header.get("source_name", "")
        if isinstance(source_name, bytes):
            source_name = source_name.decode()
        self.source_name = str(source_name)

    def _disk_frequency_slice(self, f_start: int, f_stop: int) -> tuple[slice, bool]:
        """Map internal frequency indices to an on-disk HDF5 slice.

        Args:
            f_start: Inclusive internal frequency-channel start index.
            f_stop: Exclusive internal frequency-channel stop index.

        Returns:
            Tuple of on-disk slice and whether the resulting data need reversal.
        """
        if not 0 <= f_start <= f_stop <= self.fchans:
            raise IndexError("frequency region is outside the file bounds")
        if self.ascending:
            return slice(f_start, f_stop), False
        return slice(self.fchans - f_stop, self.fchans - f_start), True

    def read_region(self,
                    t_start: int,
                    t_stop: int,
                    f_start: int,
                    f_stop: int) -> np.ndarray:
        """Read a time/frequency region from the HDF5 dataset.

        Args:
            t_start: Inclusive time-bin start index.
            t_stop: Exclusive time-bin stop index.
            f_start: Inclusive frequency-channel start index.
            f_stop: Exclusive frequency-channel stop index.

        Returns:
            Two-dimensional array in `Frame` frequency orientation.
        """
        if not 0 <= t_start <= t_stop <= self.tchans:
            raise IndexError("time region is outside the file bounds")
        freq_slice, reverse = self._disk_frequency_slice(f_start, f_stop)
        if self._dataset.ndim == 3:
            data = self._dataset[t_start:t_stop, 0, freq_slice]
        else:
            data = self._dataset[t_start:t_stop, freq_slice]
        data = np.asarray(data)
        if reverse:
            data = data[:, ::-1]
        return data

    def write_region(self,
                     t_start: int,
                     f_start: int,
                     data: np.ndarray) -> None:
        """Write a time/frequency region to the HDF5 dataset.

        Args:
            t_start: Inclusive time-bin start index.
            f_start: Inclusive frequency-channel start index.
            data: Two-dimensional data block in `Frame` orientation.
        """
        if not self.writable:
            raise OSError("spectrogram backend is read-only")
        data = np.asarray(data)
        if data.ndim != 2:
            raise ValueError("file-backed writes require a two-dimensional array")
        t_stop = t_start + data.shape[0]
        f_stop = f_start + data.shape[1]
        if not 0 <= t_start <= t_stop <= self.tchans:
            raise IndexError("time region is outside the file bounds")
        freq_slice, reverse = self._disk_frequency_slice(f_start, f_stop)
        disk_data = data[:, ::-1] if reverse else data
        if self._dataset.ndim == 3:
            self._dataset[t_start:t_stop, 0, freq_slice] = disk_data
        else:
            self._dataset[t_start:t_stop, freq_slice] = disk_data

    def flush(self) -> None:
        """Flush pending HDF5 writes."""
        self._h5.flush()

    def close(self) -> None:
        """Close the HDF5 file handle."""
        if self._h5:
            self._h5.close()

    def __getstate__(self) -> dict[str, Any]:
        state = copy.copy(self.__dict__)
        state["_h5"] = None
        state["_dataset"] = None
        return state
