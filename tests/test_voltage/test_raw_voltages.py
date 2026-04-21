import pytest
import copy
import numpy as np
import builtins

from astropy import units as u
import setigen as stg
from setigen.voltage._backend.headers import (
    _HEADER_KEY_BACKEND,
    _HEADER_KEY_BLOCSIZE,
    _HEADER_KEY_DIRECTIO,
    _HEADER_KEY_NBITS,
    _HEADER_KEY_NPOL,
    _HEADER_KEY_OBSBW,
    _HEADER_KEY_OBSFREQ,
    _HEADER_KEY_OBSERVER,
    _HEADER_KEY_OBSNCHAN,
    _HEADER_KEY_OBS_MODE,
    _HEADER_KEY_PKTIDX,
    _HEADER_KEY_PKTFMT,
    _HEADER_KEY_PKTSTART,
    _HEADER_KEY_PKTSTOP,
    _HEADER_KEY_SCANLEN,
    _HEADER_KEY_SRC_NAME,
    _HEADER_KEY_TBIN,
    _HEADER_KEY_TELESCOP,
    _HEADER_VALUE_ENCODED_1SFA_FORMAT,
    _HEADER_VALUE_ENCODED_GBT_TELESCOPE,
    _HEADER_VALUE_ENCODED_GUPPI_BACKEND,
    _HEADER_VALUE_ENCODED_RAW_MODE,
    _header_populate_configuration,
)
from setigen.voltage._backend.pipeline import _plan_subblocks
from setigen.voltage._backend.orchestration import _build_record_header
from setigen.voltage._backend.recording import _RecordConfig


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


@pytest.fixture()
def backend_setup(antenna_setup, elements_setup):
    antenna = copy.deepcopy(antenna_setup)
    digitizer, filterbank, requantizer = copy.deepcopy(elements_setup)

    num_taps = filterbank.num_taps
    num_pols = antenna.num_pols
    num_chans = 64
    block_size = num_taps * num_chans * 2 * num_pols
    return stg.voltage.RawVoltageBackend(antenna,
                                         digitizer=digitizer,
                                         filterbank=filterbank,
                                         requantizer=requantizer,
                                         start_chan=0,
                                         num_chans=num_chans,
                                         block_size=block_size,
                                         blocks_per_file=128,
                                         num_subblocks=32)


def test_build_record_header_contains_expected_core_fields(backend_setup):
    backend = copy.deepcopy(backend_setup)
    backend.num_blocks = 2
    backend.obs_length = backend.num_blocks * backend.time_per_block

    record_config = _RecordConfig.from_values(num_blocks=backend.num_blocks,
                                              length_mode="num_blocks",
                                              header_dict={"HELLO": "test_value"},
                                              verbose=False)
    header_dict = _build_record_header(backend, record_config)

    required_keys = {
        _HEADER_KEY_BACKEND,
        _HEADER_KEY_TELESCOP,
        _HEADER_KEY_OBSERVER,
        _HEADER_KEY_SRC_NAME,
        _HEADER_KEY_DIRECTIO,
        _HEADER_KEY_OBS_MODE,
        _HEADER_KEY_PKTFMT,
        _HEADER_KEY_NBITS,
        _HEADER_KEY_NPOL,
        _HEADER_KEY_BLOCSIZE,
        _HEADER_KEY_SCANLEN,
        _HEADER_KEY_TBIN,
        _HEADER_KEY_OBSNCHAN,
        _HEADER_KEY_OBSBW,
        _HEADER_KEY_OBSFREQ,
        _HEADER_KEY_PKTIDX,
        _HEADER_KEY_PKTSTART,
        _HEADER_KEY_PKTSTOP,
        "HELLO",
    }
    assert required_keys.issubset(header_dict)
    assert header_dict[_HEADER_KEY_BACKEND] == _HEADER_VALUE_ENCODED_GUPPI_BACKEND
    assert header_dict[_HEADER_KEY_OBS_MODE] == _HEADER_VALUE_ENCODED_RAW_MODE
    assert header_dict[_HEADER_KEY_PKTFMT] == _HEADER_VALUE_ENCODED_1SFA_FORMAT
    assert header_dict["HELLO"] == "test_value"


def test_build_record_header_does_not_require_template_file(monkeypatch, backend_setup):
    backend = copy.deepcopy(backend_setup)
    backend.num_blocks = 2
    backend.obs_length = backend.num_blocks * backend.time_per_block

    original_open = builtins.open

    def fail_template_open(*args, **kwargs):
        if args and str(args[0]).endswith("header_template.txt"):
            raise AssertionError("header construction should not require opening header_template.txt")
        return original_open(*args, **kwargs)

    monkeypatch.setattr(builtins, "open", fail_template_open)

    record_config = _RecordConfig.from_values(num_blocks=backend.num_blocks,
                                              length_mode="num_blocks",
                                              verbose=False)
    header_dict = _build_record_header(backend, record_config)
    assert header_dict[_HEADER_KEY_BACKEND] == _HEADER_VALUE_ENCODED_GUPPI_BACKEND
    assert header_dict[_HEADER_KEY_TELESCOP] == _HEADER_VALUE_ENCODED_GBT_TELESCOPE


def test_header_populate_configuration_preserves_core_backend_values(backend_setup):
    backend = copy.deepcopy(backend_setup)
    backend.num_blocks = 3
    backend.obs_length = backend.num_blocks * backend.time_per_block

    header_dict = _header_populate_configuration(backend, {_HEADER_KEY_PKTIDX: "16"})

    assert header_dict[_HEADER_KEY_NBITS] == backend.num_bits
    assert header_dict[_HEADER_KEY_NPOL] == backend.num_pols
    assert header_dict[_HEADER_KEY_BLOCSIZE] == backend.block_size
    assert header_dict[_HEADER_KEY_SCANLEN] == backend.obs_length
    assert header_dict[_HEADER_KEY_OBSNCHAN] == backend.num_chans * backend.num_antennas
    assert header_dict[_HEADER_KEY_PKTIDX] == 16
    assert header_dict[_HEADER_KEY_PKTSTART] == 16
    assert header_dict[_HEADER_KEY_PKTSTOP] == 16 + backend.num_blocks * backend.samples_per_block


def test_raw_record_obs_length(antenna_setup,
                               elements_setup,
                               tmp_path):
    antenna = copy.deepcopy(antenna_setup)
    digitizer, filterbank, requantizer = copy.deepcopy(elements_setup)

    num_taps = filterbank.num_taps
    num_pols = antenna.num_pols
    num_chans = 64
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

    raw_stem = tmp_path / 'example_obs_length'
    raw_path = f"{raw_stem}.0000.raw"
    rvb.record(output_file_stem=raw_stem,
               obs_length=rvb.time_per_block * 2.1,
               length_mode='obs_length',
               verbose=False)

    assert stg.voltage.get_blocks_in_file(raw_path) == 2


def test_raw_record_length_mode_errors(antenna_setup,
                                       elements_setup,
                                       tmp_path):
    antenna = copy.deepcopy(antenna_setup)
    digitizer, filterbank, requantizer = copy.deepcopy(elements_setup)

    num_taps = filterbank.num_taps
    num_pols = antenna.num_pols
    num_chans = 64
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

    with pytest.raises(ValueError, match="Value not given for 'obs_length'."):
        rvb.record(output_file_stem=tmp_path / 'missing_obs_length',
                   length_mode='obs_length',
                   verbose=False)

    with pytest.raises(ValueError, match="Value not given for 'num_blocks'."):
        rvb.record(output_file_stem=tmp_path / 'missing_num_blocks',
                   length_mode='num_blocks',
                   verbose=False)

    with pytest.raises(ValueError, match="Invalid option given for 'length_mode'."):
        rvb.record(output_file_stem=tmp_path / 'invalid_length_mode',
                   length_mode='not_a_mode',
                   verbose=False)


def test_raw_record_input_num_blocks_fallback(antenna_setup,
                                              elements_setup,
                                              tmp_path):
    antenna = copy.deepcopy(antenna_setup)
    digitizer, filterbank, requantizer = copy.deepcopy(elements_setup)

    num_taps = filterbank.num_taps
    num_pols = antenna.num_pols
    num_chans = 64
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
    rvb.input_num_blocks = 3

    raw_stem = tmp_path / 'example_input_num_blocks_fallback'
    raw_path = f"{raw_stem}.0000.raw"
    rvb.record(output_file_stem=raw_stem,
               length_mode='obs_length',
               verbose=False)

    assert stg.voltage.get_blocks_in_file(raw_path) == 3


def test_raw_record_splits_multiple_files(antenna_setup,
                                          elements_setup,
                                          tmp_path):
    antenna = copy.deepcopy(antenna_setup)
    digitizer, filterbank, requantizer = copy.deepcopy(elements_setup)

    num_taps = filterbank.num_taps
    num_pols = antenna.num_pols
    num_chans = 64
    block_size = num_taps * num_chans * 2 * num_pols
    rvb = stg.voltage.RawVoltageBackend(antenna,
                                        digitizer=digitizer,
                                        filterbank=filterbank,
                                        requantizer=requantizer,
                                        start_chan=0,
                                        num_chans=num_chans,
                                        block_size=block_size,
                                        blocks_per_file=2,
                                        num_subblocks=32)

    raw_stem = tmp_path / 'example_multi_file_split'
    rvb.record(output_file_stem=raw_stem,
               num_blocks=3,
               length_mode='num_blocks',
               verbose=False)

    assert stg.voltage.get_blocks_in_file(f"{raw_stem}.0000.raw") == 2
    assert stg.voltage.get_blocks_in_file(f"{raw_stem}.0001.raw") == 1


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
    
    num_taps = filterbank.num_taps
    num_pols = antenna.num_pols
    num_chans = 64
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
    
    raw_stem = tmp_path / 'example_1block'
    raw_path = f"{raw_stem}.0000.raw"
    rvb.record(output_file_stem=raw_stem,
               num_blocks=2, 
               length_mode='num_blocks',
               header_dict={'HELLO': 'test_value',
                            'TELESCOP': 'GBT'},
               verbose=False)
    
    header_dict = stg.voltage.read_header(raw_path)
    assert header_dict['CHAN_BW'] == '2.9296875'
    assert header_dict['OBSBW'] == '187.5'
    assert header_dict['OBSFREQ'] == '6092.28515625'
    assert header_dict['TBIN'] == '3.41333333333333E-07'
    assert header_dict['BLOCSIZE'] == '2048'
    assert header_dict['HELLO'] == 'test_value'
    assert stg.voltage.get_blocks_in_file(raw_path) == 2

    raw_params = stg.voltage.get_raw_params(raw_stem)
    assert raw_params['block_size'] == 2048
    assert raw_params['chan_bw'] == 2929687.5
    assert raw_params['num_antennas'] == 1
    assert raw_params['obs_length'] == pytest.approx(5.461333333333333e-06)

    assert stg.voltage.get_blocks_per_file(raw_stem) == 2
    assert stg.voltage.get_total_blocks(raw_stem) == 2

    assert stg.voltage.get_block_size(num_antennas=1,
                                      tchans_per_block=8,
                                      num_bits=8,
                                      num_pols=2,
                                      num_branches=1024,
                                      num_chans=64,
                                      fftlength=1,
                                      int_factor=1) == 2048

    assert stg.voltage.get_total_obs_num_samples(obs_length=None, 
                                                 num_blocks=2, 
                                                 length_mode='num_blocks',
                                                 num_antennas=1,
                                                 sample_rate=3e9,
                                                 block_size=2048,
                                                 num_bits=8,
                                                 num_pols=2,
                                                 num_branches=1024,
                                                 num_chans=64) == 16384

    # Test dist plots 
    stg.voltage.raw_utils.get_dists(raw_path, show=False)
    
    # Reduce data
    wf_data = stg.voltage.get_waterfall_from_raw(raw_path,
                                                 block_size=block_size,
                                                 num_chans=num_chans,
                                                 int_factor=1,
                                                 fftlength=1)
    
    assert wf_data.shape == (8, 64)


def test_collect_data_block_direct(antenna_setup,
                                   elements_setup):
    antenna = copy.deepcopy(antenna_setup)
    digitizer, filterbank, requantizer = copy.deepcopy(elements_setup)

    num_taps = filterbank.num_taps
    num_pols = antenna.num_pols
    num_chans = 64
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

    antenna.x.add_noise(v_mean=0, v_std=1)
    antenna.y.add_noise(v_mean=0, v_std=1)

    voltages = rvb.collect_data_block(digitize=True,
                                      requantize=False,
                                      verbose=False)

    assert voltages.shape == (num_chans, block_size // num_chans)
    assert np.isfinite(voltages).all()
    assert voltages.dtype == np.float32


def test_collect_data_block_memory_budget_increases_subblocks(antenna_setup,
                                                              elements_setup):
    antenna = copy.deepcopy(antenna_setup)
    digitizer, filterbank, requantizer = copy.deepcopy(elements_setup)

    block_size = stg.voltage.get_block_size(num_antennas=1,
                                            tchans_per_block=8,
                                            num_bits=8,
                                            num_pols=antenna.num_pols,
                                            num_branches=filterbank.num_branches,
                                            num_chans=1,
                                            fftlength=262144,
                                            int_factor=1)
    rvb = stg.voltage.RawVoltageBackend(antenna,
                                        digitizer=digitizer,
                                        filterbank=filterbank,
                                        requantizer=requantizer,
                                        start_chan=0,
                                        num_chans=1,
                                        block_size=block_size,
                                        blocks_per_file=16,
                                        num_subblocks=1,
                                        max_working_set_bytes=512 * 1024**2)

    plan = _plan_subblocks(rvb, obsnchan=rvb.num_chans * rvb.num_antennas)

    assert plan.num_subblocks > 1
    assert plan.num_subblocks >= rvb.num_subblocks


def test_raw_injection_no_directio(antenna_setup, 
                                   elements_setup,
                                   tmp_path):
    antenna = copy.deepcopy(antenna_setup)
    digitizer, filterbank, requantizer = copy.deepcopy(elements_setup)
    
    num_taps = filterbank.num_taps
    num_pols = antenna.num_pols
    num_chans = 64
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
    
    # Test not using default obs template without zero-padding / directio
    raw_stem = tmp_path / 'example_1block'
    raw_path = f"{raw_stem}.0000.raw"
    rvb.record(output_file_stem=raw_stem,
               num_blocks=2, 
               length_mode='num_blocks',
               header_dict={'HELLO': 'test_value',
                            'TELESCOP': 'GBT'},
               load_template=False,
               verbose=True)
    
    header_dict = stg.voltage.read_header(raw_path)
    assert header_dict['TELESCOP'] == 'GBT     '
    assert header_dict['OBSERVER'] == 'SETIGEN '
    assert header_dict['SRC_NAME'] == 'SYNTHETIC'
    assert header_dict['HELLO'] == 'test_value'

    raw_params = stg.voltage.get_raw_params(input_file_stem=raw_stem,
                                            start_chan=0)

    antenna = stg.voltage.Antenna(sample_rate=rvb.sample_rate,
                                  **raw_params)
    read_rvb = stg.voltage.RawVoltageBackend.from_data(input_file_stem=raw_stem,
                                                       antenna_source=antenna,
                                                       digitizer=digitizer,
                                                       filterbank=filterbank,
                                                       start_chan=0,
                                                       num_subblocks=32)
    assert read_rvb.sample_rate == rvb.sample_rate
    assert read_rvb.block_size == rvb.block_size
    assert read_rvb.num_chans == rvb.num_chans
    assert read_rvb.blocks_per_file == 2
    assert read_rvb.tbin == pytest.approx(rvb.tbin)
    assert read_rvb.samples_per_block == rvb.samples_per_block
    assert read_rvb.header_size == 80 * (len(read_rvb.input_header_dict) + 1)

    read_raw_stem = tmp_path / 'example_1block_read'
    read_raw_path = f"{read_raw_stem}.0000.raw"
    read_rvb.record(output_file_stem=read_raw_stem,
                    header_dict={'HELLO': 'test_value'},
                    load_template=False,
                    verbose=False)
    
    header_dict = stg.voltage.read_header(read_raw_path)
    assert header_dict['TELESCOP'] == 'GBT_SETIGEN'
    assert header_dict['OBSERVER'] == 'SETIGEN '
    assert header_dict['SRC_NAME'] == 'SYNTHETIC'
    assert header_dict['HELLO'] == 'test_value'
    
     # Reduce data
    wf_data = stg.voltage.get_waterfall_from_raw(read_raw_path,
                                                 block_size=block_size,
                                                 num_chans=num_chans,
                                                 int_factor=1,
                                                 fftlength=1)
    
    assert wf_data.shape == (8, 64)


def test_get_waterfall_from_raw_respects_fftlength_argument(antenna_setup,
                                                            elements_setup,
                                                            tmp_path):
    antenna = copy.deepcopy(antenna_setup)
    digitizer, filterbank, requantizer = copy.deepcopy(elements_setup)

    num_taps = filterbank.num_taps
    num_pols = antenna.num_pols
    num_chans = 64
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

    antenna.x.add_noise(v_mean=0, v_std=1)
    antenna.y.add_noise(v_mean=0, v_std=1)

    raw_stem = tmp_path / 'example_fftlength_waterfall'
    raw_path = f"{raw_stem}.0000.raw"
    rvb.record(output_file_stem=raw_stem,
               num_blocks=1,
               length_mode='num_blocks',
               verbose=False)

    wf_data = stg.voltage.get_waterfall_from_raw(raw_path,
                                                 block_size=block_size,
                                                 num_chans=num_chans,
                                                 int_factor=1,
                                                 fftlength=4)

    assert wf_data.shape == (2, 256)


def test_collect_data_block_requires_requantize_for_input_raw(antenna_setup,
                                                              elements_setup,
                                                              tmp_path):
    antenna = copy.deepcopy(antenna_setup)
    digitizer, filterbank, requantizer = copy.deepcopy(elements_setup)

    num_taps = filterbank.num_taps
    num_chans = 64
    num_pols = antenna.num_pols
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
    antenna.x.add_noise(v_mean=0, v_std=1)
    antenna.y.add_noise(v_mean=0, v_std=1)

    raw_stem = tmp_path / 'example_collect_data_block_guard'
    rvb.record(output_file_stem=raw_stem,
               num_blocks=1,
               length_mode='num_blocks',
               verbose=False)

    raw_params = stg.voltage.get_raw_params(input_file_stem=raw_stem,
                                            start_chan=0)
    read_antenna = stg.voltage.Antenna(sample_rate=rvb.sample_rate,
                                       **raw_params)
    read_rvb = stg.voltage.RawVoltageBackend.from_data(input_file_stem=raw_stem,
                                                       antenna_source=read_antenna,
                                                       digitizer=digitizer,
                                                       filterbank=filterbank,
                                                       start_chan=0,
                                                       num_subblocks=32)

    read_rvb.input_file_handler = open(f"{raw_stem}.0000.raw", "rb")
    try:
        with pytest.raises(ValueError, match="Must set 'requantize=True' when using input RAW data!"):
            read_rvb.collect_data_block(requantize=False, verbose=False)
    finally:
        read_rvb.input_file_handler.close()


def test_get_total_obs_num_samples_obs_length():
    assert stg.voltage.get_total_obs_num_samples(obs_length=5.461333333333333e-06,
                                                 length_mode='obs_length',
                                                 num_antennas=1,
                                                 sample_rate=3e9,
                                                 block_size=2048,
                                                 num_bits=8,
                                                 num_pols=2,
                                                 num_branches=1024,
                                                 num_chans=64) == 16384


def test_get_total_obs_num_samples_length_mode_errors():
    with pytest.raises(ValueError, match="Value not given for 'obs_length'."):
        stg.voltage.get_total_obs_num_samples(length_mode='obs_length')

    with pytest.raises(ValueError, match="Invalid option given for 'length_mode'."):
        stg.voltage.get_total_obs_num_samples(length_mode='not_a_mode')


def test_raw_injection_directio(antenna_setup, 
                                elements_setup,
                                tmp_path):
    antenna = copy.deepcopy(antenna_setup)
    digitizer, filterbank, requantizer = copy.deepcopy(elements_setup)
    
    num_taps = filterbank.num_taps
    num_pols = antenna.num_pols
    num_chans = 64
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
    
    # Test not using default obs template with zero-padding / directio
    raw_stem = tmp_path / 'example_1block'
    rvb.record(output_file_stem=raw_stem,
               num_blocks=2, 
               length_mode='num_blocks',
               header_dict={'HELLO': 'test_value'},
               load_template=True,
               verbose=False)

    raw_params = stg.voltage.get_raw_params(input_file_stem=raw_stem,
                                            start_chan=0)

    antenna = stg.voltage.Antenna(sample_rate=rvb.sample_rate,
                                  **raw_params)
    read_rvb = stg.voltage.RawVoltageBackend.from_data(input_file_stem=raw_stem,
                                                       antenna_source=antenna,
                                                       digitizer=digitizer,
                                                       filterbank=filterbank,
                                                       start_chan=0,
                                                       num_subblocks=32)
    assert read_rvb.sample_rate == rvb.sample_rate
    assert read_rvb.block_size == rvb.block_size
    assert read_rvb.num_chans == rvb.num_chans
    assert read_rvb.blocks_per_file == 2
    assert read_rvb.tbin == pytest.approx(rvb.tbin)
    assert read_rvb.samples_per_block == rvb.samples_per_block
    assert read_rvb.header_size == int(512 * np.ceil((80 * (len(read_rvb.input_header_dict) + 1)) / 512))

    read_raw_stem = tmp_path / 'example_1block_read'
    read_raw_path = f"{read_raw_stem}.0000.raw"
    read_rvb.record(output_file_stem=read_raw_stem,
                    header_dict={'HELLO': 'test_value'},
                    verbose=False)
    
     # Reduce data
    wf_data = stg.voltage.get_waterfall_from_raw(read_raw_path,
                                                 block_size=block_size,
                                                 num_chans=num_chans,
                                                 int_factor=1,
                                                 fftlength=1)
    
    assert wf_data.shape == (8, 64)


def test_4bit_multiantenna(antenna_array_setup,
                           tmp_path):
    antenna_array = copy.deepcopy(antenna_array_setup)
    for stream in antenna_array.bg_streams:
        stream.add_noise(0, 1)
    for antenna in antenna_array.antennas:
        for stream in antenna.streams:
            stream.add_noise(0, 1)

    num_taps = 8
    num_branches = 1024
    
    digitizers = [[stg.voltage.RealQuantizer(target_fwhm=8, 
                                             num_bits=8)
                   for pol in range(antenna_array.num_pols)]
                  for antenna in antenna_array.antennas]
    filterbanks = [[stg.voltage.PolyphaseFilterbank(num_taps=num_taps, 
                                                    num_branches=num_branches)
                    for pol in range(antenna_array.num_pols)]
                   for antenna in antenna_array.antennas]
    requantizers = [[stg.voltage.ComplexQuantizer(target_fwhm=8,
                                                  num_bits=4)
                     for pol in range(antenna_array.num_pols)]
                    for antenna in antenna_array.antennas]

    num_pols = antenna_array.num_pols
    num_chans = 64
    block_size = num_taps * num_chans * 2 * num_pols * antenna_array.num_antennas
    rvb = stg.voltage.RawVoltageBackend(antenna_array,
                                        digitizer=digitizers,
                                        filterbank=filterbanks,
                                        requantizer=requantizers,
                                        start_chan=0,
                                        num_chans=num_chans,
                                        block_size=block_size,
                                        blocks_per_file=128,
                                        num_subblocks=32)

    assert rvb.num_antennas == 3
    assert rvb.num_bits == 4

    raw_stem = tmp_path / 'example_multiantenna4bit'
    raw_path = f"{raw_stem}.0000.raw"
    rvb.record(output_file_stem=raw_stem,
               num_blocks=2, 
               length_mode='num_blocks',
               header_dict={'HELLO': 'test_value'},
               verbose=False)
    
    header_dict = stg.voltage.read_header(raw_path)
    assert header_dict['NANTS'] == '3'
    assert header_dict['NBITS'] == '4'

    # Read back multiantenna file
    read_rvb = stg.voltage.RawVoltageBackend.from_data(input_file_stem=raw_stem,
                                                       antenna_source=antenna_array,
                                                       digitizer=digitizers,
                                                       filterbank=filterbanks,
                                                       start_chan=0,
                                                       num_subblocks=32)
    assert read_rvb.num_antennas == 3
    assert read_rvb.num_bits == 4

    read_raw_stem = tmp_path / 'example_multiantenna4bit_read'
    read_rvb.record(output_file_stem=read_raw_stem,
                    header_dict={'HELLO': 'test_value'},
                    verbose=False)
