from __future__ import annotations

import os
import pathlib
import struct
from typing import Any

import numpy as np
from astropy.time import Time

from .base import SpectrogramBackend


_HEADER_KEYWORD_TYPES = {
    "telescope_id": "<l",
    "machine_id": "<l",
    "data_type": "<l",
    "barycentric": "<l",
    "pulsarcentric": "<l",
    "nbits": "<l",
    "nsamples": "<l",
    "nchans": "<l",
    "nifs": "<l",
    "nbeams": "<l",
    "ibeam": "<l",
    "rawdatafile": "str",
    "source_name": "str",
    "az_start": "<d",
    "za_start": "<d",
    "tstart": "<d",
    "tsamp": "<d",
    "fch1": "<d",
    "foff": "<d",
    "refdm": "<d",
    "period": "<d",
    "src_raj": "<d",
    "src_dej": "<d",
}


def _read_keyword(handle: Any) -> tuple[str, Any]:
    """Read one SIGPROC header keyword and value.

    Args:
        handle: Binary file handle positioned at a keyword record.

    Returns:
        Tuple of keyword name and decoded value.
    """
    n_raw = handle.read(4)
    if len(n_raw) != 4:
        raise RuntimeError("Unexpected end of file while reading SIGPROC header")
    n_bytes = struct.unpack("<I", n_raw)[0]
    if n_bytes > 255:
        n_bytes = 16
    keyword = handle.read(n_bytes).decode("ascii")
    if keyword in {"HEADER_START", "HEADER_END"}:
        return keyword, None

    dtype = _HEADER_KEYWORD_TYPES[keyword]
    if dtype == "<l":
        return keyword, struct.unpack(dtype, handle.read(4))[0]
    if dtype == "<d":
        return keyword, struct.unpack(dtype, handle.read(8))[0]
    if dtype == "str":
        str_len = struct.unpack("<I", handle.read(4))[0]
        return keyword, handle.read(str_len).decode("ascii")
    raise RuntimeError(f"Unsupported SIGPROC header type: {dtype}")


def _read_header(path: pathlib.Path) -> tuple[dict[str, Any], int]:
    """Read a SIGPROC filterbank header.

    Args:
        path: Filterbank file path.

    Returns:
        Tuple of header dictionary and byte offset where data begin.
    """
    header: dict[str, Any] = {}
    with open(path, "rb") as handle:
        keyword, _ = _read_keyword(handle)
        if keyword != "HEADER_START":
            raise RuntimeError("Not a valid SIGPROC filterbank file")
        while True:
            keyword, value = _read_keyword(handle)
            if keyword == "HEADER_END":
                return header, handle.tell()
            header[keyword] = value


def _dtype_from_nbits(nbits: int) -> np.dtype:
    """Resolve a NumPy dtype from a SIGPROC sample width.

    Args:
        nbits: Bits per sample from the filterbank header.

    Returns:
        NumPy dtype for on-disk samples.
    """
    if nbits == 32:
        return np.dtype("<f4")
    if nbits == 16:
        return np.dtype("<u2")
    if nbits == 8:
        return np.dtype("u1")
    raise ValueError(f"Unsupported filterbank sample width: {nbits} bits")


class FilSpectrogramBackend(SpectrogramBackend):
    """Region reader/writer for SIGPROC filterbank files."""

    def __init__(self, path: pathlib.Path, mode: str = "r") -> None:
        """Open a filterbank spectrogram backend.

        Args:
            path: Filterbank file path.
            mode: Backend open mode.
        """
        self.path = path
        self.mode = mode
        self.header, self._data_offset = _read_header(path)
        self._initialize_metadata()
        file_mode = "r+b" if self.writable else "rb"
        self._file = open(path, file_mode)

    def _initialize_metadata(self) -> None:
        """Populate frame-compatible metadata from the filterbank header."""
        self.fchans = int(self.header["nchans"])
        self.nifs = int(self.header.get("nifs", 1))
        self.dtype = _dtype_from_nbits(int(self.header["nbits"]))
        self._sample_bytes = self.dtype.itemsize
        data_bytes = os.path.getsize(self.path) - self._data_offset
        row_bytes = self.fchans * self.nifs * self._sample_bytes
        if row_bytes <= 0 or data_bytes % row_bytes != 0:
            raise ValueError("filterbank data size is inconsistent with its header")
        self.tchans = data_bytes // row_bytes
        self.shape = (int(self.tchans), int(self.fchans))
        self.df = abs(float(self.header["foff"])) * 1e6
        self.dt = float(self.header["tsamp"])
        self.ascending = float(self.header["foff"]) > 0
        self.fch1 = float(self.header["fch1"]) * 1e6
        self.t_start = Time(float(self.header["tstart"]), format="mjd").unix
        source_name = self.header.get("source_name", "")
        if isinstance(source_name, bytes):
            source_name = source_name.decode()
        self.source_name = str(source_name)

    def _disk_frequency_slice(self, f_start: int, f_stop: int) -> tuple[int, int, bool]:
        """Map internal frequency indices to on-disk channel bounds.

        Args:
            f_start: Inclusive internal frequency-channel start index.
            f_stop: Exclusive internal frequency-channel stop index.

        Returns:
            Tuple of on-disk start, on-disk stop, and whether to reverse data.
        """
        if not 0 <= f_start <= f_stop <= self.fchans:
            raise IndexError("frequency region is outside the file bounds")
        if self.ascending:
            return f_start, f_stop, False
        return self.fchans - f_stop, self.fchans - f_start, True

    def _row_offset(self, time_index: int, if_id: int, f_start: int) -> int:
        """Calculate the byte offset for a row window.

        Args:
            time_index: Time-bin index.
            if_id: IF index.
            f_start: On-disk frequency-channel start index.

        Returns:
            Byte offset from the start of the file.
        """
        sample_index = (time_index * self.nifs + if_id) * self.fchans + f_start
        return self._data_offset + sample_index * self._sample_bytes

    def read_region(self,
                    t_start: int,
                    t_stop: int,
                    f_start: int,
                    f_stop: int) -> np.ndarray:
        """Read a time/frequency region from the filterbank file.

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
        disk_start, disk_stop, reverse = self._disk_frequency_slice(f_start, f_stop)
        out = np.empty((t_stop - t_start, f_stop - f_start), dtype=self.dtype)
        for local_t, time_index in enumerate(range(t_start, t_stop)):
            self._file.seek(self._row_offset(time_index, 0, disk_start))
            row = np.fromfile(self._file, dtype=self.dtype, count=disk_stop - disk_start)
            if row.shape[0] != disk_stop - disk_start:
                raise RuntimeError("Unexpected end of file while reading filterbank data")
            out[local_t] = row[::-1] if reverse else row
        return out

    def write_region(self,
                     t_start: int,
                     f_start: int,
                     data: np.ndarray) -> None:
        """Write a time/frequency region to the filterbank file.

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
        disk_start, disk_stop, reverse = self._disk_frequency_slice(f_start, f_stop)
        if disk_stop - disk_start != data.shape[1]:
            raise ValueError("write region width does not match data width")
        for local_t, time_index in enumerate(range(t_start, t_stop)):
            row = data[local_t, ::-1] if reverse else data[local_t]
            self._file.seek(self._row_offset(time_index, 0, disk_start))
            np.asarray(row, dtype=self.dtype).tofile(self._file)

    def flush(self) -> None:
        """Flush pending filterbank writes to disk."""
        self._file.flush()
        os.fsync(self._file.fileno())

    def close(self) -> None:
        """Close the filterbank file handle."""
        if not self._file.closed:
            self._file.close()
