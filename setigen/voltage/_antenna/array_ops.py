from __future__ import annotations

from typing import Any
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _coerce_delays(*, delays: list[int] | tuple[int, ...] | NDArray[np.integer] | None,
                   num_antennas: int,
                   xp: Any) -> tuple[NDArray[np.integer] | np.ndarray, int]:
    """Normalize antenna delays into an integer array.

    Args:
        delays: Optional sequence of per-antenna delays in samples.
        num_antennas: Number of antennas in the array.
        xp: NumPy- or CuPy-like array module.

    Returns:
        Tuple of normalized delay array and maximum delay.

    Raises:
        ValueError: If the number of provided delays does not match the number
            of antennas.
    """
    if delays is None:
        delays_array = xp.zeros(num_antennas, dtype=int)
    else:
        if len(delays) != num_antennas:
            raise ValueError("delays must match num_antennas")
        delays_array = xp.array(delays).astype(int)
    return delays_array, int(xp.max(delays_array))


def _set_antenna_time(antenna: Any, t: float) -> None:
    """Reset an antenna and its streams to a new observation start time.

    Args:
        antenna: Antenna object to update.
        t: New start time in seconds.
    """
    antenna.start_obs = True
    antenna.t_start = t
    antenna.x.set_time(t)
    if antenna.num_pols == 2:
        antenna.y.set_time(t)


def _get_single_antenna_samples(antenna: Any, num_samples: int, *, xp: Any) -> np.ndarray:
    """Collect one antenna's current foreground samples.

    Args:
        antenna: Antenna object to sample.
        num_samples: Number of time samples to generate.
        xp: NumPy- or CuPy-like array module.

    Returns:
        Array of generated samples shaped for the backend pipeline.
    """
    if antenna.num_pols == 2:
        samples = [[antenna.x.get_samples(num_samples),
                    antenna.y.get_samples(num_samples)]]
    else:
        samples = [[antenna.x.get_samples(num_samples)]]

    antenna.t_start += num_samples * antenna.dt
    antenna.start_obs = False

    return xp.array(samples)


def _reset_array_time_state(array_obj: Any, t: float) -> None:
    """Reset array-level time state before a new observation.

    Args:
        array_obj: Multi-antenna array object to reset.
        t: New start time in seconds.
    """
    array_obj.start_obs = True
    array_obj.t_start = t
    array_obj.bg_x.set_time(t)
    if array_obj.num_pols == 2:
        array_obj.bg_y.set_time(t)
    for antenna in array_obj.antennas:
        antenna.bg_cache = [None, None]
        antenna.set_time(t)


def _populate_background_streams(array_obj: Any, num_samples: int) -> int:
    """Advance shared background streams for an array observation step.

    Args:
        array_obj: Multi-antenna array object to advance.
        num_samples: Number of foreground samples requested.

    Returns:
        Number of background samples actually generated, including any delay
        margin needed at observation start.
    """
    if array_obj.start_obs:
        bg_num_samples = num_samples + array_obj.max_delay
    else:
        bg_num_samples = num_samples

    array_obj.bg_x.get_samples(bg_num_samples)
    if array_obj.num_pols == 2:
        array_obj.bg_y.get_samples(bg_num_samples)
    return bg_num_samples


def _apply_background_to_antenna(array_obj: Any,
                                 antenna: Any,
                                 *,
                                 bg_num_samples: int,
                                 num_samples: int,
                                 xp: Any) -> None:
    """Apply cached shared background noise to one antenna's streams.

    Args:
        array_obj: Multi-antenna array object providing background streams.
        antenna: Antenna receiving the background noise.
        bg_num_samples: Number of available background samples.
        num_samples: Number of foreground samples requested.
        xp: NumPy- or CuPy-like array module.
    """
    antenna.x.get_samples(num_samples)

    if array_obj.start_obs:
        bg_x_v = array_obj.bg_x.v[array_obj.max_delay - antenna.delay:bg_num_samples - antenna.delay]
    else:
        bg_x_v = xp.concatenate([antenna.bg_cache[0], array_obj.bg_x.v])[:bg_num_samples]

    antenna.bg_cache[0] = array_obj.bg_x.v[bg_num_samples - antenna.delay:]
    antenna.x.v += bg_x_v

    if array_obj.num_pols == 2:
        antenna.y.get_samples(num_samples)

        if array_obj.start_obs:
            bg_y_v = array_obj.bg_y.v[array_obj.max_delay - antenna.delay:bg_num_samples - antenna.delay]
        else:
            bg_y_v = xp.concatenate([antenna.bg_cache[1], array_obj.bg_y.v])[:bg_num_samples]

        antenna.bg_cache[1] = array_obj.bg_y.v[bg_num_samples - antenna.delay:]
        antenna.y.v += bg_y_v


def _collect_array_samples(array_obj: Any, *, xp: Any) -> np.ndarray:
    """Collect the latest per-antenna samples into one array.

    Args:
        array_obj: Multi-antenna array object containing sampled antennas.
        xp: NumPy- or CuPy-like array module.

    Returns:
        Array of samples shaped for backend ingestion.
    """
    if array_obj.num_pols == 2:
        samples = [[antenna.x.v, antenna.y.v] for antenna in array_obj.antennas]
    else:
        samples = [[antenna.x.v] for antenna in array_obj.antennas]
    return xp.array(samples)
