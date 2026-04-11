from __future__ import annotations


def _coerce_delays(*, delays, num_antennas, xp):
    if delays is None:
        delays_array = xp.zeros(num_antennas, dtype=int)
    else:
        if len(delays) != num_antennas:
            raise ValueError("delays must match num_antennas")
        delays_array = xp.array(delays).astype(int)
    return delays_array, int(xp.max(delays_array))


def _set_antenna_time(antenna, t):
    antenna.start_obs = True
    antenna.t_start = t
    antenna.x.set_time(t)
    if antenna.num_pols == 2:
        antenna.y.set_time(t)


def _get_single_antenna_samples(antenna, num_samples, *, xp):
    if antenna.num_pols == 2:
        samples = [[antenna.x.get_samples(num_samples),
                    antenna.y.get_samples(num_samples)]]
    else:
        samples = [[antenna.x.get_samples(num_samples)]]

    antenna.t_start += num_samples * antenna.dt
    antenna.start_obs = False

    return xp.array(samples)


def _reset_array_time_state(array_obj, t):
    array_obj.start_obs = True
    array_obj.t_start = t
    array_obj.bg_x.set_time(t)
    if array_obj.num_pols == 2:
        array_obj.bg_y.set_time(t)
    for antenna in array_obj.antennas:
        antenna.bg_cache = [None, None]
        antenna.set_time(t)


def _populate_background_streams(array_obj, num_samples):
    if array_obj.start_obs:
        bg_num_samples = num_samples + array_obj.max_delay
    else:
        bg_num_samples = num_samples

    array_obj.bg_x.get_samples(bg_num_samples)
    if array_obj.num_pols == 2:
        array_obj.bg_y.get_samples(bg_num_samples)
    return bg_num_samples


def _apply_background_to_antenna(array_obj, antenna, *, bg_num_samples, num_samples, xp):
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


def _collect_array_samples(array_obj, *, xp):
    if array_obj.num_pols == 2:
        samples = [[antenna.x.v, antenna.y.v] for antenna in array_obj.antennas]
    else:
        samples = [[antenna.x.v] for antenna in array_obj.antennas]
    return xp.array(samples)
