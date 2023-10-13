import pytest
import copy
import numpy as np

from astropy import units as u
import setigen as stg
import blimpy as bl


@pytest.fixture()
def antenna_setup():
    sample_rate = 3e9
    antenna = stg.voltage.Antenna(sample_rate=sample_rate, 
                                  fch1=6*u.GHz,
                                  ascending=True,
                                  num_pols=2)
    return antenna


@pytest.fixture()
def antenna_array_setup():
    sample_rate = 3e9

    delays = np.array([0, 100, 200])
    antenna_array = stg.voltage.MultiAntennaArray(num_antennas=3,
                                                  sample_rate=sample_rate,
                                                  fch1=6*u.GHz,
                                                  ascending=False,
                                                  num_pols=2,
                                                  delays=delays)
    return antenna_array


@pytest.fixture()
def elements_setup():
    num_taps = 8
    num_branches = 1024
    
    digitizer = stg.voltage.RealQuantizer(target_fwhm=32,
                                          num_bits=8)

    filterbank = stg.voltage.PolyphaseFilterbank(num_taps=num_taps, 
                                                 num_branches=num_branches)

    requantizer = stg.voltage.ComplexQuantizer(target_fwhm=32,
                                               num_bits=8)
    
    return digitizer, filterbank, requantizer


def test_noise_injection(antenna_setup):
    antenna = copy.deepcopy(antenna_setup)
    for stream in antenna.streams:
        stream.add_noise(0, 1)
        assert stream.noise_std == 1
    
    samples = antenna.get_samples(10000)
    for pol_samples in samples[0]:
        assert abs(np.mean(pol_samples)) < 0.1
        assert abs(np.std(pol_samples) - 1) < 0.1
        
        
def test_noise_injection_array(antenna_array_setup):
    antenna_array = copy.deepcopy(antenna_array_setup)
    for stream in antenna_array.bg_streams:
        stream.add_noise(0, 1)
    for antenna in antenna_array.antennas:
        for stream in antenna.streams:
            assert stream.bg_noise_std == 1
            stream.add_noise(0, 1)
            assert stream.noise_std == 1
            assert stream.get_total_noise_std() == pytest.approx(2**0.5)
            
    samples = antenna_array.get_samples(10000)
    for antenna_samples in samples:
        for pol_samples in antenna_samples:
            assert abs(np.mean(pol_samples)) < 0.1
            assert abs(np.std(pol_samples) - 2**0.5) < 0.1
    

def test_raw_creation(antenna_setup, 
                      elements_setup,
                      tmp_path):
    antenna = copy.deepcopy(antenna_setup)
    digitizer, filterbank, requantizer = copy.deepcopy(elements_setup)
    
    num_taps = 8
    num_branches = 1024
    num_chans = 64
    num_pols = 2
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
    antenna.x.add_noise(v_mean=0, 
                        v_std=1)
    antenna.y.add_noise(v_mean=0, 
                        v_std=1)
    antenna.x.add_constant_signal(f_start=6002.2e6, 
                                  drift_rate=-2*u.Hz/u.s, 
                                  level=0.002)
    antenna.y.add_constant_signal(f_start=6002.2e6, 
                                  drift_rate=-2*u.Hz/u.s, 
                                  level=0.002,
                                  phase=np.pi/2)
    
    rvb.record(output_file_stem=tmp_path / 'example_1block',
               num_blocks=1, 
               length_mode='num_blocks',
               header_dict={'HELLO': 'test_value',
                            'TELESCOP': 'GBT'},
               verbose=False)
    
    # Check out header
    header_dict = {}
    with open(tmp_path / 'example_1block.0000.raw', "rb") as f:
        i = 1
        chunk = f.read(80)
        while True:
            key, item = chunk.decode().split('=')
            header_dict[key.strip()] = item.strip().strip("''")

            chunk = f.read(80)
            if f"{'END':<80}".encode() in chunk:
                break
            i += 1
    
    assert header_dict['CHAN_BW'] == '2.9296875'
    assert header_dict['OBSBW'] == '187.5'
    assert header_dict['OBSFREQ'] == '6092.28515625'
    assert header_dict['TBIN'] == '3.41333333333333E-07'
    assert header_dict['BLOCSIZE'] == '2048'
    assert header_dict['HELLO'] == 'test_value'
    
    # Reduce data
    wf_data = stg.voltage.get_waterfall_from_raw(tmp_path / 'example_1block.0000.raw',
                                                 block_size=block_size,
                                                 num_chans=num_chans,
                                                 int_factor=1,
                                                 fftlength=1)
    
    assert wf_data.shape == (8, 64)
