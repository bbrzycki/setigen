import pytest
import numpy as np

import setigen as stg


def test_dedrift():
    for ascending in [False, True]:
        for sign in [1, -1]:
            frame = stg.Frame(shape=(16, 512), ascending=ascending, seed=0)   
            drift_rate = sign * frame.unit_drift_rate 
            frame.add_constant_signal(frame.get_frequency(64),
                                      drift_rate=drift_rate,
                                      level=1,
                                      width=frame.df,
                                      f_profile_type="box")
            frame.add_metadata({"drift_rate": drift_rate})

            dd_frame = stg.dedrift(frame)
            assert np.max(stg.integrate(dd_frame)) == pytest.approx(1)

            dd_frame = stg.dedrift(frame, drift_rate=drift_rate)
            assert np.max(stg.integrate(dd_frame)) == pytest.approx(1)


def test_dedrift_requires_rate_when_metadata_missing():
    frame = stg.Frame(shape=(16, 512), seed=0)

    with pytest.raises(KeyError, match="Please specify a drift rate"):
        stg.dedrift(frame)


def test_dedrift_rejects_rate_that_exceeds_frame_width():
    frame = stg.Frame(shape=(16, 512), seed=0)
    excessive_drift_rate = frame.unit_drift_rate * frame.fchans

    with pytest.raises(ValueError, match="too high for the frame dimensions"):
        stg.dedrift(frame, drift_rate=excessive_drift_rate)
