def get_slice(fr, l, r):
    """
    Slice frame data with left and right index bounds.
    
    Parameters
    ----------
    fr : Frame
        Input frame
    l : int
        Left bound
    r : int
        Right bound
        
    Returns
    -------
    s_fr : Frame
        Sliced frame
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
                        waterfall=fr.check_waterfall(),
                        seed=fr.rng)

    return s_fr
