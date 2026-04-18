from __future__ import annotations

import numpy as np


def _decode_raw_block(data_chunk,
                      *,
                      num_bits,
                      num_pols,
                      num_chans,
                      start_chan=0,
                      num_selected_chans=None):
    if num_selected_chans is None:
        num_selected_chans = num_chans - start_chan

    stop_chan = start_chan + num_selected_chans
    if start_chan < 0 or num_selected_chans <= 0 or stop_chan > num_chans:
        raise ValueError("Selected coarse channel range is out of bounds for this RAW input.")

    rawbuffer = np.frombuffer(data_chunk, dtype=np.int8).reshape((num_chans, -1))
    rawbuffer = rawbuffer[start_chan:stop_chan]

    if num_bits == 8:
        num_samples = rawbuffer.shape[1] // (2 * num_pols)
        voltages = np.empty((num_samples, num_selected_chans, num_pols), dtype=np.complex64)
        for pol in range(num_pols):
            base = 2 * pol
            real_vals = rawbuffer[:, base::2 * num_pols]
            imag_vals = rawbuffer[:, (base + 1)::2 * num_pols]
            voltages[:, :, pol] = (real_vals + 1j * imag_vals).T
        return voltages

    if num_bits == 4:
        num_samples = rawbuffer.shape[1] // num_pols
        voltages = np.empty((num_samples, num_selected_chans, num_pols), dtype=np.complex64)
        for pol in range(num_pols):
            q_vals = rawbuffer[:, pol::num_pols]
            real_vals = q_vals // 16
            imag_vals = q_vals - 16 * real_vals
            imag_vals = imag_vals.copy()
            imag_vals[imag_vals >= 8] -= 16
            voltages[:, :, pol] = (real_vals + 1j * imag_vals).T
        return voltages

    raise ValueError(f"{num_bits} bits not supported.")
