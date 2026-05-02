from __future__ import annotations

from typing import TYPE_CHECKING

from ._frame.context import _finalize_derived_frame, _source_bounds_metadata

if TYPE_CHECKING:
    from .frame import Frame


def get_slice(fr: Frame, l: int, r: int) -> Frame:
    """Return a frequency slice of a frame.

    Args:
        fr: Input frame.
        l: Left frequency index.
        r: Right frequency index.

    Returns:
        Sliced frame.
    """
    s_data = fr.data[:, l:r]

    # Match frequency to truncated frame
    if fr.ascending:
        fch1 = fr.fs[l]
    else:
        fch1 = fr.fs[r - 1]

    s_fr = fr.from_data(fr.df, 
                        fr.dt, 
                        fch1, 
                        fr.ascending,
                        s_data,
                        seed=fr.rng,
                        t_start=fr.t_start,
                        source_name=fr.source_name)
    _finalize_derived_frame(
        fr,
        s_fr,
        operation="slice",
        product_type="frame",
        source_bounds=_source_bounds_metadata(
            fr,
            f_index_range=(l, r),
            t_index_range=(0, fr.tchans),
        ),
    )

    return s_fr
