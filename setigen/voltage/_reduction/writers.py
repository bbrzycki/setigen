from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import shutil
from types import SimpleNamespace

import h5py
import numpy as np

from blimpy import Waterfall
from blimpy.io import sigproc


@lru_cache(maxsize=1)
def _get_base_filterbank_header():
    sample_path = Path(__file__).resolve().parents[2] / "assets" / "sample.fil"
    waterfall = Waterfall(str(sample_path), load_data=False)
    return dict(waterfall.header)


def _build_filterbank_header(input_spec, metadata, *, pol_mode):
    header = _get_base_filterbank_header().copy()
    header["source_name"] = input_spec.source_name
    header["rawdatafile"] = input_spec.rawdatafile
    header["tstart"] = input_spec.tstart_mjd
    header["tsamp"] = metadata.dt_s
    header["nchans"] = metadata.total_fchans
    header["nifs"] = metadata.nifs
    header["nbits"] = 32
    header["data_type"] = 1
    signed_foff = metadata.df_hz * 1e-6 if metadata.ascending else -metadata.df_hz * 1e-6
    header["foff"] = signed_foff
    header["fch1"] = metadata.fch1_hz * 1e-6
    if pol_mode == 1:
        header["nifs"] = 1
    return header


def _prepare_output_path(output_path, *, overwrite, tmp_dir=None):
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output file '{output_path}' already exists.")

    if tmp_dir is None:
        return output_path, output_path

    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / output_path.name
    if tmp_path.exists():
        tmp_path.unlink()
    return output_path, tmp_path


def _finalize_output(final_path, write_path):
    if write_path != final_path:
        shutil.move(str(write_path), str(final_path))


@dataclass
class _FilWriter:
    final_path: Path
    write_path: Path
    header: dict
    _handle: object = None

    def __enter__(self):
        self._handle = open(self.write_path, "wb")
        self._handle.write(sigproc.generate_sigproc_header(SimpleNamespace(header=self.header)))
        return self

    def append(self, chunk):
        np.asarray(chunk, dtype=np.float32).tofile(self._handle)

    def __exit__(self, exc_type, exc, tb):
        if self._handle is not None:
            self._handle.close()
        if exc_type is None:
            _finalize_output(self.final_path, self.write_path)
        elif self.write_path.exists():
            self.write_path.unlink()
        return False


@dataclass
class _H5Writer:
    final_path: Path
    write_path: Path
    header: dict
    total_fchans: int
    nifs: int
    _handle: object = None
    _data: object = None
    _mask: object = None

    def __enter__(self):
        self._handle = h5py.File(self.write_path, "w")
        self._handle.attrs["CLASS"] = "FILTERBANK"
        self._handle.attrs["VERSION"] = "1.0"
        self._data = self._handle.create_dataset(
            "data",
            shape=(0, self.nifs, self.total_fchans),
            maxshape=(None, self.nifs, self.total_fchans),
            dtype=np.float32,
        )
        self._mask = self._handle.create_dataset(
            "mask",
            shape=(0, self.nifs, self.total_fchans),
            maxshape=(None, self.nifs, self.total_fchans),
            dtype=np.uint8,
        )
        self._data.dims[2].label = b"frequency"
        self._data.dims[1].label = b"feed_id"
        self._data.dims[0].label = b"time"
        self._mask.dims[2].label = b"frequency"
        self._mask.dims[1].label = b"feed_id"
        self._mask.dims[0].label = b"time"
        for key, value in self.header.items():
            self._data.attrs[key] = value
        return self

    def append(self, chunk):
        chunk = np.asarray(chunk, dtype=np.float32)
        start = self._data.shape[0]
        stop = start + chunk.shape[0]
        self._data.resize(stop, axis=0)
        self._mask.resize(stop, axis=0)
        self._data[start:stop] = chunk
        self._mask[start:stop] = 0

    def __exit__(self, exc_type, exc, tb):
        if self._handle is not None:
            self._handle.close()
        if exc_type is None:
            _finalize_output(self.final_path, self.write_path)
        elif self.write_path.exists():
            self.write_path.unlink()
        return False


def _create_writer(output_path, *, output_format, overwrite, tmp_dir, header, total_fchans, nifs):
    final_path, write_path = _prepare_output_path(output_path, overwrite=overwrite, tmp_dir=tmp_dir)
    if output_format == "fil":
        return _FilWriter(final_path=final_path, write_path=write_path, header=header)
    if output_format == "h5":
        return _H5Writer(final_path=final_path,
                         write_path=write_path,
                         header=header,
                         total_fchans=total_fchans,
                         nifs=nifs)
    raise ValueError(f"Unsupported output format '{output_format}'.")
