import numpy as np
from .frame import Frame


def dedrift(fr, drift_rate=None):
    """
    Dedrift frame with a provided drift rate, or with the "drift_rate"
    keyword in the frame's metadata. This function dedrifts with respect
    to the center of the frame, so signals at the edges may get cut off.
    
    Parameters
    ----------
    fr : Frame
        Input frame
    drift_rate : float, optional
        Drift rate in Hz/s
        
    Returns
    -------
    dr_fr : Frame
        De-drifted frame
    """
    if drift_rate is None:
        if 'drift_rate' in fr.metadata:
            drift_rate = fr.metadata['drift_rate']
        else:
            raise KeyError('Please specify a drift rate to account for')
            
    # Calculate maximum pixel offset and raise an exception if necessary
    max_offset = int(abs(drift_rate) * fr.tchans * fr.dt / fr.df)
    if max_offset >= fr.fchans:
        raise ValueError(f'The provided drift rate ({drift_rate:.2f} Hz/s) ' 
                         f'is too high for the frame dimensions')
    tr_data = np.zeros((fr.data.shape[0], fr.data.shape[1] - max_offset))

    for i in range(fr.tchans):
        offset = int(abs(drift_rate) * i * fr.dt / fr.df)
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
        
    dd_fr = Frame.from_data(fr.df, 
                            fr.dt, 
                            fch1, 
                            fr.ascending,
                            tr_data,
                            metadata=fr.metadata,
                            waterfall=fr.check_waterfall())
#     if dd_fr.waterfall is not None and 'source_name' in dd_fr.waterfall.header:
#         dd_fr.waterfall.header['source_name'] += '_dedrifted'
    return dd_fr
