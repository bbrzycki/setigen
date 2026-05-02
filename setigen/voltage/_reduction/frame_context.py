from __future__ import annotations

from typing import Any

from astropy.time import Time

from .writers import _build_filterbank_header


def _frame_context_kwargs(input_spec: Any, metadata: Any, *, pol_mode: int) -> dict[str, Any]:
    """Build `Frame.from_data` context from voltage reduction metadata.

    Args:
        input_spec: Parsed or synthetic RAW input description.
        metadata: Derived reduction metadata.
        pol_mode: Polarization mode for the source product.

    Returns:
        Keyword arguments that preserve filterbank context on a `Frame`.
    """
    header = _build_filterbank_header(input_spec, metadata, pol_mode=pol_mode)
    source_name = header.get("source_name")
    if isinstance(source_name, bytes):
        source_name = source_name.decode()

    kwargs = {"header": header}
    if source_name is not None:
        kwargs["source_name"] = source_name
    if header.get("tstart") is not None:
        kwargs["t_start"] = Time(header["tstart"], format="mjd").unix
    return kwargs
