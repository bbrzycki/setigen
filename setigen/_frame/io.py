from __future__ import annotations

import pathlib
import pickle
from typing import Any

import numpy as np

from blimpy import Waterfall
from blimpy.io import sigproc


def _create_synthetic_waterfall(frame: Any, *, max_load: int = 1) -> Waterfall:
    """Create a synthetic waterfall container aligned with a frame.

    Args:
        frame: Frame instance providing shape and header metadata.
        max_load: Maximum load parameter for the template waterfall.

    Returns:
        Synthetic waterfall object configured for the frame.
    """
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


def _update_waterfall(
    frame: Any,
    *,
    filename: str | pathlib.Path | None = None,
    max_load: int = 1,
) -> None:
    """Synchronize a frame's attached waterfall with its current data.

    Args:
        frame: Frame instance to synchronize.
        filename: Optional output filename to attach to the waterfall container.
        max_load: Maximum load parameter for a lazily created waterfall.
    """
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


def _encode_bytestrings(frame: Any) -> None:
    """Encode waterfall header string fields as bytes for blimpy writes.

    Args:
        frame: Frame instance whose waterfall header should be encoded.
    """
    for key in ["source_name", "rawdatafile"]:
        if key in frame.waterfall.header and not isinstance(frame.waterfall.header[key], bytes):
            frame.waterfall.header[key] = frame.waterfall.header[key].encode()


def _decode_bytestrings(frame: Any) -> None:
    """Decode waterfall header byte fields back to strings.

    Args:
        frame: Frame instance whose waterfall header should be decoded.
    """
    for key in ["source_name", "rawdatafile"]:
        if key in frame.waterfall.header and isinstance(frame.waterfall.header[key], bytes):
            frame.waterfall.header[key] = frame.waterfall.header[key].decode()


def _get_waterfall(frame: Any) -> Waterfall:
    """Return an up-to-date waterfall representation for a frame.

    Args:
        frame: Frame instance to convert.

    Returns:
        Updated waterfall object.
    """
    _update_waterfall(frame)
    return frame.waterfall


def _check_waterfall(frame: Any) -> Waterfall | None:
    """Return an updated waterfall when one is attached to the frame.

    Args:
        frame: Frame instance to inspect.

    Returns:
        Updated waterfall object, or `None` when no waterfall is attached.
    """
    if frame.waterfall is None:
        return None
    return _get_waterfall(frame)


def _save_fil(frame: Any, filename: str | pathlib.Path, *, max_load: int = 1) -> None:
    """Write a frame to SIGPROC filterbank format.

    Args:
        frame: Frame instance to serialize.
        filename: Output `.fil` path.
        max_load: Maximum load parameter for a lazily created waterfall.
    """
    _update_waterfall(frame, filename=filename, max_load=max_load)
    _encode_bytestrings(frame)
    frame.waterfall.write_to_fil(filename)
    _decode_bytestrings(frame)


def _save_hdf5(frame: Any, filename: str | pathlib.Path, *, max_load: int = 1) -> None:
    """Write a frame to HDF5 waterfall format.

    Args:
        frame: Frame instance to serialize.
        filename: Output `.h5` path.
        max_load: Maximum load parameter for a lazily created waterfall.
    """
    _update_waterfall(frame, filename=filename, max_load=max_load)
    _encode_bytestrings(frame)
    frame.waterfall.write_to_hdf5(filename)
    _decode_bytestrings(frame)


def _save_npy(frame: Any, filename: str | pathlib.Path) -> None:
    """Write frame data to a NumPy binary file.

    Args:
        frame: Frame instance to serialize.
        filename: Output `.npy` path.
    """
    np.save(filename, frame.data)


def _load_npy(frame: Any, filename: str | pathlib.Path) -> None:
    """Load frame data from a NumPy binary file.

    Args:
        frame: Frame instance to update.
        filename: Input `.npy` path.
    """
    frame.data = np.load(filename)


def _save_pickle(frame: Any, filename: str | pathlib.Path) -> None:
    """Serialize a frame with pickle.

    Args:
        frame: Frame instance to serialize.
        filename: Output pickle path.
    """
    with open(filename, "wb") as f:
        pickle.dump(frame, f)


def _load_pickle(filename: str | pathlib.Path) -> Any:
    """Deserialize a pickled frame.

    Args:
        filename: Input pickle path.

    Returns:
        Deserialized frame object.
    """
    with open(filename, "rb") as f:
        return pickle.load(f)
