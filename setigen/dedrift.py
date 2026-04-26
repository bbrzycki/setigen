from __future__ import annotations

import numpy as np
from .frame import Frame


def dedrift(fr: Frame, drift_rate: float | None = None) -> Frame:
    """De-drift a frame using an explicit or metadata-provided drift rate.

    This operation aligns drifting signals relative to the center of the
    frame. Signals near the frequency edges may be truncated by the resulting
    shift.

    Args:
        fr: Input frame to de-drift.
        drift_rate: Drift rate in Hz/s. When omitted, the function looks for
            ``"drift_rate"`` in ``fr.metadata``.

    Returns:
        New frame with the drift removed.

    Raises:
        KeyError: If no drift rate is provided or stored in the frame
            metadata.
        ValueError: If the requested drift rate would shift the signal beyond
            the frame width.
    """
    if drift_rate is None:
        if 'drift_rate' in fr.metadata:
            drift_rate = fr.metadata['drift_rate']
        else:
            raise KeyError('Please specify a drift rate to account for')
            
    # Calculate maximum pixel offset and raise an exception if necessary
    max_offset = int(np.round(abs(drift_rate) * fr.tchans * fr.dt / fr.df))
    if max_offset >= fr.fchans:
        raise ValueError(f'The provided drift rate ({drift_rate:.2f} Hz/s) ' 
                         f'is too high for the frame dimensions')
    tr_data = np.zeros((fr.data.shape[0], fr.data.shape[1] - max_offset))

    for i in range(fr.tchans):
        offset = int(np.round(abs(drift_rate) * i * fr.dt / fr.df))
        if drift_rate >= 0:
            start_idx = 0 + offset
            end_idx = start_idx + tr_data.shape[1]
        else:
            end_idx = fr.data.shape[1] - offset
            start_idx = end_idx - tr_data.shape[1]
        tr_data[i] = fr.data[i, start_idx:end_idx]
        
    # Match frequency to truncated fr
    if fr.ascending:
        if drift_rate >= 0:
            fch1 = fr.fs[0]
        else:
            fch1 = fr.fs[max_offset]
    else:
        if drift_rate >= 0:
            fch1 = fr.fs[::-1][max_offset]
        else:
            fch1 = fr.fs[::-1][0]
        
    dd_fr = fr.from_data(fr.df, 
                         fr.dt, 
                         fch1, 
                         fr.ascending,
                         tr_data,
                         metadata=fr.metadata,
                         seed=fr.rng,
                         t_start=fr.t_start,
                         source_name=fr.source_name,
                         header=fr.header)
#     if dd_fr.waterfall is not None and 'source_name' in dd_fr.waterfall.header:
#         dd_fr.waterfall.header['source_name'] += '_dedrifted'
    return dd_fr
