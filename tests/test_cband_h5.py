import pytest
from pathlib import Path

import numpy as np
import setigen as stg


@pytest.fixture()
def frame_setup_from_h5():
    path = Path(__file__).resolve().parent / "assets/sample.fil"
    frame = stg.Frame(waterfall=path)
    return frame


def test_setup_from_h5(frame_setup_from_h5):
    frame = frame_setup_from_h5
    assert frame.fchans == 1024
    assert frame.tchans == 32
    assert frame.shape == (32, 1024)

    assert frame.df == pytest.approx(1.3969838619232178)
    assert frame.dt == pytest.approx(1.431655765333332)
    assert frame.fmax == pytest.approx(6663999999.873341)
    assert frame.fmin == pytest.approx(6663998569.361866)

    assert frame.mean == pytest.approx(484170.38)
    assert frame.noise_mean == pytest.approx(461739.75)

    assert frame.std == pytest.approx(253477.6)
    assert frame.noise_std == pytest.approx(217443.1)

    assert np.min(frame.data) == pytest.approx(15350.96)
