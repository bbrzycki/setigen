from __future__ import annotations

import copy
from typing import Any

from astropy.time import Time


_DERIVED_METADATA_EXCLUDE_KEYS = {"file_backed", "path"}


def _copy_custom_metadata(source: Any, target: Any) -> None:
    """Copy non-core metadata between frame-like objects.

    Args:
        source: Original frame-like object.
        target: Derived frame-like object to update.
    """
    metadata = copy.deepcopy(getattr(source, "metadata", {}))
    try:
        for key, value in source.get_params().items():
            if metadata.get(key) == value:
                metadata.pop(key)
    except AttributeError:
        pass
    for key in _DERIVED_METADATA_EXCLUDE_KEYS:
        metadata.pop(key, None)
    if metadata:
        target.add_metadata(metadata)


def _frame_header_fields(frame: Any) -> dict[str, Any]:
    """Return header fields that must match a frame's current geometry.

    Args:
        frame: Frame-like object providing observing metadata.

    Returns:
        Header fields synchronized to the frame.
    """
    return {
        "source_name": frame.source_name,
        "tsamp": frame.dt,
        "tstart": Time(frame.t_start, format="unix").mjd,
        "nchans": frame.fchans,
        "nifs": 1,
        "fch1": frame.fch1 * 1e-6,
        "foff": frame.df * (1 if frame.ascending else -1) * 1e-6,
    }


def _copy_header_context(source: Any, target: Any) -> None:
    """Copy and synchronize observational header context.

    Args:
        source: Original frame-like object.
        target: Derived frame-like object to update.
    """
    source_header = getattr(source, "header", None)
    target_header = getattr(target, "header", None)
    if source_header is None and target_header is None:
        return

    if source_header is not None:
        target.header = copy.deepcopy(source_header)
    elif target_header is not None:
        target.header = copy.deepcopy(target_header)
    target.header.update(_frame_header_fields(target))


def _source_bounds_metadata(
    source: Any,
    *,
    f_index_range: tuple[int, int] | None = None,
    t_index_range: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Build serializable bounds metadata for a derived frame.

    Args:
        source: Source frame-like object.
        f_index_range: Half-open source frequency index range.
        t_index_range: Half-open source time index range.

    Returns:
        Bounds metadata describing the source region.
    """
    f_start, f_stop = f_index_range or (0, source.fchans)
    t_start, t_stop = t_index_range or (0, source.tchans)
    bounds: dict[str, Any] = {
        "time_index_range": (int(t_start), int(t_stop)),
        "frequency_index_range": (int(f_start), int(f_stop)),
    }

    if hasattr(source, "time_edges"):
        bounds["time_range_s"] = (
            float(source.time_edges[t_start]),
            float(source.time_edges[t_stop]),
        )
    if hasattr(source, "frequency_edges"):
        bounds["frequency_range_hz"] = (
            float(source.frequency_edges[f_start]),
            float(source.frequency_edges[f_stop]),
        )
    return bounds


def _finalize_derived_frame(
    source: Any,
    target: Any,
    *,
    operation: str,
    product_type: str | None = None,
    collapsed_axis: str | None = None,
    reducer: str | None = None,
    normalization: str | None = None,
    source_bounds: dict[str, Any] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> None:
    """Attach source context and operation provenance to a derived frame.

    Args:
        source: Original frame-like object.
        target: Derived frame-like object to update.
        operation: Operation that produced the target.
        product_type: Optional product kind such as ``"spectrum"``.
        collapsed_axis: Optional collapsed axis name.
        reducer: Optional reducer name.
        normalization: Optional normalization policy.
        source_bounds: Optional source-region metadata.
        extra_metadata: Optional additional operation metadata.
    """
    _copy_custom_metadata(source, target)
    _copy_header_context(source, target)

    derived = {
        "operation": operation,
        "source_shape": tuple(getattr(source, "shape", ())),
    }
    if product_type is not None:
        derived["product_type"] = product_type
    if collapsed_axis is not None:
        derived["collapsed_axis"] = collapsed_axis
    if reducer is not None:
        derived["reducer"] = reducer
    if normalization is not None:
        derived["normalization"] = normalization
    if source_bounds is not None:
        derived["source_bounds"] = copy.deepcopy(source_bounds)
    if extra_metadata:
        derived.update(copy.deepcopy(extra_metadata))
    target.add_metadata({"derived": derived})


def _copy_frame_context(source: Any, target: Any) -> None:
    """Copy non-shape observational context between frame-like objects.

    Args:
        source: Original frame-like object.
        target: Derived frame-like object to update.
    """
    _copy_custom_metadata(source, target)
    _copy_header_context(source, target)
