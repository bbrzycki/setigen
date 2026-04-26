from __future__ import annotations

from typing import TYPE_CHECKING

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
                        metadata=fr.metadata,
                        seed=fr.rng,
                        t_start=fr.t_start,
                        source_name=fr.source_name,
                        header=fr.header)

    return s_fr
