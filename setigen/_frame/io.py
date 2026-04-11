from __future__ import annotations

import pathlib
import pickle

import numpy as np

from blimpy import Waterfall
from blimpy.io import sigproc


def _create_synthetic_waterfall(frame, *, max_load=1):
    path = pathlib.Path(__file__).resolve().parents[1] / "assets" / "sample.fil"
    waterfall = Waterfall(str(path), max_load=max_load)
    waterfall.header["source_name"] = frame.source_name
    waterfall.header["rawdatafile"] = "Synthetic"

    container_attr = {
        "t_begin": 0,
        "t_end": frame.tchans,
        "file_size_bytes": frame.tchans * frame.fchans * waterfall.header["nbits"] / 8,
        "n_channels_in_file": frame.fchans,
        "n_ints_in_file": frame.tchans,
        "file_shape": (frame.tchans, 1, frame.fchans),
        "f_end": frame.fmax * 1e-6,
        "f_begin": frame.fmin * 1e-6,
        "f_stop": frame.fmax * 1e-6,
        "f_start": frame.fmin * 1e-6,
        "t_start": 0,
        "t_stop": frame.tchans,
        "selection_shape": (frame.tchans, 1, frame.fchans),
        "chan_start_idx": 0,
        "chan_stop_idx": frame.fchans,
    }
    for key, value in container_attr.items():
        setattr(waterfall.container, key, value)

    wat_attr = {
        "n_channels_in_file": frame.fchans,
        "n_ints_in_file": frame.tchans,
        "file_shape": (frame.tchans, 1, frame.fchans),
        "file_size_bytes": frame.tchans * frame.fchans * waterfall.header["nbits"] / 8,
        "selection_shape": (frame.tchans, 1, frame.fchans),
    }
    for key, value in wat_attr.items():
        setattr(waterfall, key, value)

    return waterfall


def _update_waterfall(frame, *, filename=None, max_load=1):
    if frame.waterfall is None:
        frame.waterfall = _create_synthetic_waterfall(frame, max_load=max_load)

    frame.waterfall.data = frame.data[:, np.newaxis, :]
    if not frame.ascending:
        frame.waterfall.data = frame.waterfall.data[:, :, ::-1]

    header_attr = {
        "tsamp": frame.dt,
        "tstart": frame.mjd,
        "nchans": frame.fchans,
        "fch1": frame.fch1 * 1e-6,
    }
    if frame.ascending:
        header_attr["foff"] = frame.df * 1e-6
    else:
        header_attr["foff"] = frame.df * -1e-6
    frame.waterfall.header.update(header_attr)
    frame.waterfall.file_header.update(header_attr)

    if filename is not None:
        frame.waterfall.container.filename = str(pathlib.Path(filename).resolve())
    frame.waterfall.container.idx_data = len(sigproc.generate_sigproc_header(frame.waterfall))


def _encode_bytestrings(frame):
    for key in ["source_name", "rawdatafile"]:
        if key in frame.waterfall.header and not isinstance(frame.waterfall.header[key], bytes):
            frame.waterfall.header[key] = frame.waterfall.header[key].encode()


def _decode_bytestrings(frame):
    for key in ["source_name", "rawdatafile"]:
        if key in frame.waterfall.header and isinstance(frame.waterfall.header[key], bytes):
            frame.waterfall.header[key] = frame.waterfall.header[key].decode()


def _get_waterfall(frame):
    _update_waterfall(frame)
    return frame.waterfall


def _check_waterfall(frame):
    if frame.waterfall is None:
        return None
    return _get_waterfall(frame)


def _save_fil(frame, filename, *, max_load=1):
    _update_waterfall(frame, filename=filename, max_load=max_load)
    _encode_bytestrings(frame)
    frame.waterfall.write_to_fil(filename)
    _decode_bytestrings(frame)


def _save_hdf5(frame, filename, *, max_load=1):
    _update_waterfall(frame, filename=filename, max_load=max_load)
    _encode_bytestrings(frame)
    frame.waterfall.write_to_hdf5(filename)
    _decode_bytestrings(frame)


def _save_npy(frame, filename):
    np.save(filename, frame.data)


def _load_npy(frame, filename):
    frame.data = np.load(filename)


def _save_pickle(frame, filename):
    with open(filename, "wb") as f:
        pickle.dump(frame, f)


def _load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)
