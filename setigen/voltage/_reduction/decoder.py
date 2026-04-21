from __future__ import annotations

import numpy as np


def _decode_raw_block(data_chunk: bytes,
                      *,
                      num_bits: int,
                      num_pols: int,
                      num_chans: int,
                      start_chan: int = 0,
                      num_selected_chans: int | None = None) -> np.ndarray:
    """Decode one RAW block into complex coarse-channel voltages.

    Args:
        data_chunk: Raw byte payload from a single RAW block.
        num_bits: Bit depth per complex component. Supported values are 4 and
            8.
        num_pols: Number of polarizations encoded in the payload.
        num_chans: Total number of coarse channels in the payload.
        start_chan: First coarse channel to decode.
        num_selected_chans: Number of coarse channels to decode starting at
            ``start_chan``.

    Returns:
        Complex voltage array with shape ``(time, coarse_chan, pol)``.

    Raises:
        ValueError: If the selected channel range is invalid or the bit depth
            is unsupported.
    """
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
