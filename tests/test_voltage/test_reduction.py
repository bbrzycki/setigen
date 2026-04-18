from pathlib import Path
import copy

import numpy as np
from numpy.testing import assert_allclose
import pytest
from astropy import units as u
from blimpy import Waterfall

import setigen as stg
from setigen.voltage._reduction.channelize import _channelize_block
from setigen.voltage._reduction.decoder import _decode_raw_block


def _make_backend(*,
                  num_pols=2,
                  num_bits=8,
                  num_chans=4,
                  num_branches=64,
                  fftlength=8,
                  tchans_per_block=8,
                  blocks_per_file=8):
    sample_rate = 3e9 * u.Hz
    antenna = stg.voltage.Antenna(sample_rate=sample_rate,
                                  fch1=6 * u.GHz,
                                  ascending=True,
                                  num_pols=num_pols,
                                  seed=123)
    for stream in antenna.streams:
        stream.add_noise(v_mean=0, v_std=1)

    digitizer = stg.voltage.RealQuantizer(target_fwhm=32, num_bits=8)
    filterbank = stg.voltage.PolyphaseFilterbank(num_taps=8,
                                                 num_branches=num_branches)
    requantizer = stg.voltage.ComplexQuantizer(target_fwhm=32,
                                               num_bits=num_bits)
    block_size = stg.voltage.get_block_size(num_antennas=1,
                                            tchans_per_block=tchans_per_block,
                                            num_bits=num_bits,
                                            num_pols=num_pols,
                                            num_branches=num_branches,
                                            num_chans=num_chans,
                                            fftlength=fftlength,
                                            int_factor=1)
    backend = stg.voltage.RawVoltageBackend(antenna,
                                            digitizer=digitizer,
                                            filterbank=filterbank,
                                            requantizer=requantizer,
                                            start_chan=0,
                                            num_chans=num_chans,
                                            block_size=block_size,
                                            blocks_per_file=blocks_per_file,
                                            num_subblocks=4)
    return backend


def test_decode_raw_block_8bit():
    rawbuffer = np.array([
        [1, 2, 3, 4, 5, 6, 7, 8],
        [11, 12, 13, 14, 15, 16, 17, 18],
    ], dtype=np.int8)

    voltages = _decode_raw_block(rawbuffer.tobytes(),
                                 num_bits=8,
                                 num_pols=2,
                                 num_chans=2)

    assert voltages.shape == (2, 2, 2)
    assert_allclose(voltages[:, 0, 0], np.array([1 + 2j, 5 + 6j]))
    assert_allclose(voltages[:, 0, 1], np.array([3 + 4j, 7 + 8j]))
    assert_allclose(voltages[:, 1, 0], np.array([11 + 12j, 15 + 16j]))
    assert_allclose(voltages[:, 1, 1], np.array([13 + 14j, 17 + 18j]))


def test_decode_raw_block_4bit():
    rawbuffer = np.array([
        [0x3E, 0x47],
        [0x12, 0x8F],
    ], dtype=np.uint8).view(np.int8)

    voltages = _decode_raw_block(rawbuffer.tobytes(),
                                 num_bits=4,
                                 num_pols=1,
                                 num_chans=2)

    assert voltages.shape == (2, 2, 1)
    assert_allclose(voltages[:, 0, 0], np.array([3 - 2j, 4 + 7j]))
    assert_allclose(voltages[:, 1, 0], np.array([1 + 2j, -8 - 1j]))


def test_channelize_block_polarization_modes():
    voltages = np.array([
        [[1 + 0j, 2 + 0j]],
        [[0 + 1j, 0 + 2j]],
        [[-1 + 0j, -2 + 0j]],
        [[0 - 1j, 0 - 2j]],
    ], dtype=np.complex64)

    spec_x = np.fft.fftshift(np.fft.fft(voltages[:, 0, 0].reshape(2, 2), axis=1) / 2**0.5, axes=1)
    spec_y = np.fft.fftshift(np.fft.fft(voltages[:, 0, 1].reshape(2, 2), axis=1) / 2**0.5, axes=1)
    xx = np.abs(spec_x) ** 2
    yy = np.abs(spec_y) ** 2
    xy = spec_x * np.conj(spec_y)

    total = _channelize_block(voltages,
                              fftlength=2,
                              integration_factor=1,
                              pol_mode=1,
                              backend="numpy")
    full_pol = _channelize_block(voltages,
                                 fftlength=2,
                                 integration_factor=1,
                                 pol_mode=4,
                                 backend="numpy")
    full_stokes = _channelize_block(voltages,
                                    fftlength=2,
                                    integration_factor=1,
                                    pol_mode=-4,
                                    backend="numpy")

    assert_allclose(total[:, 0, :], (xx + yy).reshape(2, 2))
    assert_allclose(full_pol[:, 0, :], xx.reshape(2, 2))
    assert_allclose(full_pol[:, 1, :], yy.reshape(2, 2))
    assert_allclose(full_pol[:, 2, :], np.real(xy).reshape(2, 2))
    assert_allclose(full_pol[:, 3, :], np.imag(xy).reshape(2, 2))
    assert_allclose(full_stokes[:, 0, :], (xx + yy).reshape(2, 2))
    assert_allclose(full_stokes[:, 1, :], (xx - yy).reshape(2, 2))
    assert_allclose(full_stokes[:, 2, :], (2 * np.real(xy)).reshape(2, 2))
    assert_allclose(full_stokes[:, 3, :], (-2 * np.imag(xy)).reshape(2, 2))


def test_reduce_raw_to_frame_matches_existing_helper(tmp_path):
    backend = _make_backend(num_pols=2, num_bits=8, num_chans=4, fftlength=8, tchans_per_block=8)
    raw_stem = tmp_path / "helper_parity"
    raw_path = tmp_path / "helper_parity.0000.raw"

    backend.record(output_file_stem=raw_stem,
                   num_blocks=1,
                   length_mode="num_blocks",
                   verbose=False)

    spec = stg.voltage.RawReductionSpec(fftlength=8,
                                        integration_factor=1,
                                        pol_mode=stg.voltage.PolarizationMode.TOTAL_POWER,
                                        output_format="fil")
    frame = stg.voltage.reduce_raw_to_frame(raw_path, spec, max_blocks=1)
    helper = stg.voltage.get_waterfall_from_raw(str(raw_path),
                                                block_size=backend.block_size,
                                                num_chans=backend.num_chans,
                                                fftlength=8,
                                                int_factor=1)

    assert frame.data.shape == helper.shape
    assert_allclose(frame.data, helper)


def test_reduce_raw_roundtrip_fil_total_power(tmp_path):
    backend = _make_backend(num_pols=2, num_bits=8, num_chans=4, fftlength=8, tchans_per_block=8)
    raw_stem = tmp_path / "roundtrip_total"
    output_path = tmp_path / "roundtrip_total.fil"

    backend.record(output_file_stem=raw_stem,
                   num_blocks=2,
                   length_mode="num_blocks",
                   verbose=False)

    spec = stg.voltage.RawReductionSpec(fftlength=8,
                                        integration_factor=1,
                                        pol_mode=1,
                                        output_format="fil")
    stg.voltage.reduce_raw(raw_stem, output_path, spec, overwrite=True)

    wf = Waterfall(str(output_path))
    frame = stg.Frame(waterfall=str(output_path))

    assert wf.header["nifs"] == 1
    assert wf.header["nchans"] == 32
    assert frame.data.shape == (16, 32)
    assert frame.fchans == 32
    assert frame.tchans == 16


def test_reduce_raw_roundtrip_h5_total_power(tmp_path):
    backend = _make_backend(num_pols=2, num_bits=8, num_chans=4, fftlength=8, tchans_per_block=8)
    raw_stem = tmp_path / "roundtrip_total_h5"
    output_path = tmp_path / "roundtrip_total.h5"

    backend.record(output_file_stem=raw_stem,
                   num_blocks=2,
                   length_mode="num_blocks",
                   verbose=False)

    spec = stg.voltage.RawReductionSpec(fftlength=8,
                                        integration_factor=1,
                                        pol_mode=1,
                                        output_format="h5")
    stg.voltage.reduce_raw(raw_stem, output_path, spec, overwrite=True)

    wf = Waterfall(str(output_path))
    frame = stg.Frame(waterfall=str(output_path))

    assert wf.header["nifs"] == 1
    assert wf.header["nchans"] == 32
    assert frame.data.shape == (16, 32)


def test_reduce_raw_roundtrip_fil_full_stokes(tmp_path):
    backend = _make_backend(num_pols=2, num_bits=8, num_chans=4, fftlength=8, tchans_per_block=8)
    raw_stem = tmp_path / "roundtrip_stokes"
    output_path = tmp_path / "roundtrip_stokes.fil"

    backend.record(output_file_stem=raw_stem,
                   num_blocks=1,
                   length_mode="num_blocks",
                   verbose=False)

    spec = stg.voltage.RawReductionSpec(fftlength=8,
                                        integration_factor=1,
                                        pol_mode=-4,
                                        output_format="fil")
    stg.voltage.reduce_raw(raw_stem, output_path, spec, overwrite=True)

    wf = Waterfall(str(output_path))
    assert wf.header["nifs"] == 4
    assert wf.header["nchans"] == 32
    assert wf.data.shape == (8, 4, 32)


def test_reduce_raw_roundtrip_h5_full_pol(tmp_path):
    backend = _make_backend(num_pols=2, num_bits=8, num_chans=4, fftlength=8, tchans_per_block=8)
    raw_stem = tmp_path / "roundtrip_full_pol"
    output_path = tmp_path / "roundtrip_full_pol.h5"

    backend.record(output_file_stem=raw_stem,
                   num_blocks=1,
                   length_mode="num_blocks",
                   verbose=False)

    spec = stg.voltage.RawReductionSpec(fftlength=8,
                                        integration_factor=1,
                                        pol_mode=4,
                                        output_format="h5")
    stg.voltage.reduce_raw(raw_stem, output_path, spec, overwrite=True)

    wf = Waterfall(str(output_path))
    assert wf.header["nifs"] == 4
    assert wf.header["nchans"] == 32
    assert wf.data.shape == (8, 4, 32)


def test_reduce_raw_multifile_stem(tmp_path):
    backend = _make_backend(num_pols=2,
                            num_bits=8,
                            num_chans=4,
                            fftlength=8,
                            tchans_per_block=8,
                            blocks_per_file=1)
    raw_stem = tmp_path / "multifile"
    output_path = tmp_path / "multifile.fil"

    backend.record(output_file_stem=raw_stem,
                   num_blocks=2,
                   length_mode="num_blocks",
                   verbose=False)

    spec = stg.voltage.RawReductionSpec(fftlength=8,
                                        integration_factor=1,
                                        pol_mode=1,
                                        output_format="fil")
    stg.voltage.reduce_raw(raw_stem, output_path, spec, overwrite=True)

    wf = Waterfall(str(output_path))
    assert wf.data.shape == (16, 1, 32)


def test_reduce_raw_detects_injected_signal_after_dedrift(tmp_path):
    backend = _make_backend(num_pols=2,
                            num_bits=8,
                            num_chans=4,
                            num_branches=64,
                            fftlength=64,
                            tchans_per_block=16)
    raw_stem = tmp_path / "signal"

    spec = stg.voltage.RawReductionSpec(fftlength=64,
                                        integration_factor=1,
                                        pol_mode=1,
                                        output_format="fil")

    unit_drift_rate = stg.voltage.get_unit_drift_rate(backend,
                                                      fftlength=spec.fftlength,
                                                      int_factor=spec.integration_factor)
    signal_level = stg.voltage.get_level(snr=12,
                                         raw_voltage_backend=backend,
                                         fftlength=spec.fftlength,
                                         num_blocks=1,
                                         length_mode="num_blocks")

    coarse_chan = backend.num_chans // 2
    fine_bw = backend.chan_bw / spec.fftlength
    coarse_center = backend.fch1 + coarse_chan * backend.chan_bw
    signal_frequency = coarse_center + (spec.fftlength // 2 - 3 - spec.fftlength / 2) * fine_bw
    drift_rate = unit_drift_rate

    for stream in backend.antenna_source.streams:
        leakage = stg.voltage.get_leakage_factor(signal_frequency,
                                                 backend,
                                                 spec.fftlength)
        level = stream.get_total_noise_std() * leakage * signal_level
        stream.add_constant_signal(f_start=signal_frequency * u.Hz,
                                   drift_rate=drift_rate * u.Hz / u.s,
                                   level=level)

    backend.record(output_file_stem=raw_stem,
                   num_blocks=1,
                   length_mode="num_blocks",
                   verbose=False)

    frame = stg.voltage.reduce_raw_to_frame(raw_stem, spec)
    dd_frame = stg.dedrift(frame, drift_rate=drift_rate)
    spectrum = stg.spectrum(dd_frame, mode="sum", normalize=True)

    peak_index = int(np.argmax(spectrum.data))
    expected_index = int(round((signal_frequency - dd_frame.fch1) / dd_frame.df))

    assert abs(peak_index - expected_index) <= 2
    assert np.max(spectrum.data) > 3
