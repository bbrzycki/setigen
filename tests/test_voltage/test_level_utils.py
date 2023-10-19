import pytest
import copy
import numpy as np
from numpy.testing import assert_allclose

from astropy import units as u
import setigen as stg


@pytest.fixture()
def backend_setup():
    sample_rate = 3e9
    num_taps = 8
    num_branches = 1024
    num_pols = 2
    num_chans = 64
    antenna = stg.voltage.Antenna(sample_rate=sample_rate, 
                                  fch1=6*u.GHz,
                                  ascending=True,
                                  num_pols=num_pols)
    digitizer = stg.voltage.RealQuantizer(target_fwhm=8,
                                          num_bits=8)
    filterbank = stg.voltage.PolyphaseFilterbank(num_taps=num_taps, 
                                                 num_branches=num_branches)
    requantizer = stg.voltage.ComplexQuantizer(target_fwhm=8,
                                               num_bits=8)
    block_size = num_taps * num_chans * 2 * num_pols
    rvb = stg.voltage.RawVoltageBackend(antenna,
                                        digitizer=digitizer,
                                        filterbank=filterbank,
                                        requantizer=requantizer,
                                        start_chan=0,
                                        num_chans=num_chans,
                                        block_size=block_size,
                                        blocks_per_file=128,
                                        num_subblocks=32)
    return rvb
    


def test_unit_drift_rate(backend_setup):
    rvb = copy.deepcopy(backend_setup)
    fftlength = 2**14
    int_factor = 2**6
    assert stg.voltage.get_unit_drift_rate(rvb, 
                                           fftlength=fftlength, 
                                           int_factor=int_factor) == pytest.approx(499.60036108132044)


def test_level(backend_setup):
    rvb = copy.deepcopy(backend_setup)
    fftlength = 2**14
    assert stg.voltage.get_level(snr=25, 
                                 raw_voltage_backend=rvb, 
                                 fftlength=fftlength, 
                                 obs_length=300) == pytest.approx(0.00013489699168632314)


def test_leakage_factor(backend_setup):
    rvb = copy.deepcopy(backend_setup)
    fftlength = 2**14
    assert stg.voltage.get_leakage_factor(f_start=6001e6, 
                                          raw_voltage_backend=rvb, 
                                          fftlength=fftlength) == pytest.approx(1.3318603467759598)