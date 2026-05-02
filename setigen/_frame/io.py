from __future__ import annotations

import copy
import pathlib
import pickle
from typing import Any

import numpy as np

from blimpy import Waterfall
from blimpy.io import sigproc


def _close_waterfall_handles(waterfall: Any) -> None:
    """Close live file handles owned by a blimpy waterfall when present.

    Args:
        waterfall: Waterfall-like object to inspect.
    """
    h5 = getattr(getattr(waterfall, "container", None), "h5", None)
    if h5 is None:
        return
    try:
        h5.close()
    except Exception:
        pass


def _has_live_h5_handle(waterfall: Any) -> bool:
    """Return whether a waterfall is backed by a live HDF5 file handle.

    Args:
        waterfall: Waterfall-like object to inspect.

    Returns:
        Whether the object owns a currently valid HDF5 file handle.
    """
    h5 = getattr(getattr(waterfall, "container", None), "h5", None)
    if h5 is None:
        return False
    try:
        return bool(h5.id.valid)
    except Exception:
        return True


def _base_filterbank_header() -> dict[str, Any]:
    """Return a packaged filterbank header template.

    Returns:
        Deep-copied SIGPROC-compatible header template.
    """
    path = pathlib.Path(__file__).resolve().parents[1] / "assets" / "sample.fil"
    waterfall = Waterfall(str(path), load_data=False)
    header = copy.deepcopy(waterfall.header)
    _close_waterfall_handles(waterfall)
    return header


def _frame_waterfall_header(frame: Any) -> dict[str, Any]:
    """Build a blimpy-compatible header from frame metadata.

    Args:
        frame: Frame instance providing scientific metadata.

    Returns:
        Header dictionary suitable for constructing a Waterfall adapter.
    """
    header = _base_filterbank_header()
    if getattr(frame, "header", None) is not None:
        header.update(copy.deepcopy(frame.header))

    header.update({
        "source_name": frame.source_name,
        "tsamp": frame.dt,
        "tstart": frame.mjd,
        "nchans": frame.fchans,
        "nifs": 1,
        "fch1": frame.fch1 * 1e-6,
        "foff": frame.df * (1 if frame.ascending else -1) * 1e-6,
    })
    header.setdefault("rawdatafile", "Synthetic")
    header.setdefault("nbits", 32)
    return header


def _frame_waterfall_data(frame: Any) -> np.ndarray:
    """Return frame data in blimpy's time/IF/frequency layout.

    Args:
        frame: Frame instance providing data and orientation metadata.

    Returns:
        Three-dimensional data array with shape ``(time, if, frequency)``.
    """
    data = frame.data[:, np.newaxis, :]
    if not frame.ascending:
        data = data[:, :, ::-1]
    return data


def _sync_waterfall_adapter(waterfall: Waterfall, frame: Any) -> Waterfall:
    """Synchronize an existing in-memory waterfall adapter with a frame.

    Args:
        waterfall: Existing in-memory Waterfall adapter.
        frame: Frame instance whose current state should be reflected.

    Returns:
        The same Waterfall adapter after metadata and data synchronization.
    """
    header = _frame_waterfall_header(frame)
    data = _frame_waterfall_data(frame)

    waterfall.header.clear()
    waterfall.header.update(header)
    waterfall.file_header = waterfall.header
    waterfall.data = data

    waterfall.n_ints_in_file = frame.tchans
    waterfall.selection_shape = data.shape
    waterfall.n_channels_in_file = frame.fchans
    waterfall.file_shape = data.shape
    waterfall.file_size_bytes = frame.tchans * frame.fchans * header["nbits"] / 8

    container = waterfall.container
    container.header = waterfall.header
    container.n_channels_in_file = frame.fchans
    container._n_bytes = int(header["nbits"] / 8)
    if header["foff"] < 0:
        container.f_end = header["fch1"]
        container.f_begin = container.f_end + frame.fchans * header["foff"]
    else:
        container.f_begin = header["fch1"]
        container.f_end = container.f_begin + frame.fchans * header["foff"]
    container.f_start = container.f_begin
    container.f_stop = container.f_end
    container.t_begin = 0
    container.t_end = frame.tchans
    container.t_start = 0
    container.t_stop = frame.tchans
    container.selection_shape = data.shape
    container.n_ints_in_file = frame.tchans
    return waterfall


def _create_synthetic_waterfall(frame: Any, *, max_load: int = 1) -> Waterfall:
    """Create a synthetic waterfall container aligned with a frame.

    Args:
        frame: Frame instance providing shape and header metadata.
        max_load: Maximum load parameter for the template waterfall.

    Returns:
        Synthetic waterfall object configured for the frame.
    """
    del max_load
    waterfall = Waterfall(header_dict=_frame_waterfall_header(frame),
                          data_array=_frame_waterfall_data(frame))
    return _sync_waterfall_adapter(waterfall, frame)


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
    if frame.waterfall is None or _has_live_h5_handle(frame.waterfall):
        frame.waterfall = _create_synthetic_waterfall(frame, max_load=max_load)
    else:
        _sync_waterfall_adapter(frame.waterfall, frame)

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
    try:
        frame.waterfall.write_to_fil(filename)
    finally:
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
    try:
        frame.waterfall.write_to_hdf5(filename)
    finally:
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
