from __future__ import annotations

from .._typing import BandpassProfile


def constant_bp_profile(level: float = 1) -> BandpassProfile:
    """Return a constant bandpass profile.

    Args:
        level: Constant bandpass level.

    Returns:
        Bandpass-profile callable.
    """
    def bp_profile(f):
        return level
    return bp_profile
