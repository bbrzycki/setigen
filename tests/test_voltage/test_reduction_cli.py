from pathlib import Path

from astropy import units as u
from click.testing import CliRunner
from blimpy import Waterfall

import setigen as stg
from setigen.voltage.cli import main


def _make_backend():
    antenna = stg.voltage.Antenna(sample_rate=3e9 * u.Hz,
                                  fch1=6 * u.GHz,
                                  ascending=True,
                                  num_pols=2,
                                  seed=123)
    for stream in antenna.streams:
        stream.add_noise(v_mean=0, v_std=1)

    digitizer = stg.voltage.RealQuantizer(target_fwhm=32, num_bits=8)
    filterbank = stg.voltage.PolyphaseFilterbank(num_taps=8, num_branches=64)
    requantizer = stg.voltage.ComplexQuantizer(target_fwhm=32, num_bits=8)
    block_size = stg.voltage.get_block_size(num_antennas=1,
                                            tchans_per_block=8,
                                            num_bits=8,
                                            num_pols=2,
                                            num_branches=64,
                                            num_chans=4,
                                            fftlength=8,
                                            int_factor=1)
    return stg.voltage.RawVoltageBackend(antenna,
                                         digitizer=digitizer,
                                         filterbank=filterbank,
                                         requantizer=requantizer,
                                         start_chan=0,
                                         num_chans=4,
                                         block_size=block_size,
                                         blocks_per_file=8,
                                         num_subblocks=4)


def test_reduction_cli_writes_filterbank(tmp_path):
    backend = _make_backend()
    raw_stem = tmp_path / "cli_reduce"
    output_path = tmp_path / "cli_reduce.fil"
    backend.record(output_file_stem=raw_stem,
                   num_blocks=1,
                   length_mode="num_blocks",
                   verbose=False)

    runner = CliRunner()
    result = runner.invoke(main, [
        str(raw_stem),
        str(output_path),
        "--fftlength", "8",
        "--integration-factor", "1",
        "--pol-mode", "1",
        "--format", "fil",
        "--overwrite",
    ])

    assert result.exit_code == 0, result.output
    assert Path(result.output.strip()) == output_path
    wf = Waterfall(str(output_path))
    assert wf.header["nifs"] == 1
    assert wf.header["nchans"] == 32
