import numpy as np
from numpy.testing import assert_allclose
import pytest
from astropy import units as u
from blimpy import Waterfall

import setigen as stg
from setigen.voltage._reduction.channelize import _channelize_block
from setigen.voltage._reduction.decoder import _decode_raw_block
from setigen.voltage.reduction import _reduce_chunks


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


def test_channelize_block_integrates_consecutive_fine_spectra():
    voltages = np.array([
        [[1 + 0j, 0 + 1j]],
        [[2 + 0j, 0 + 2j]],
        [[3 + 0j, 0 + 3j]],
        [[4 + 0j, 0 + 4j]],
        [[5 + 0j, 0 + 5j]],
        [[6 + 0j, 0 + 6j]],
        [[7 + 0j, 0 + 7j]],
        [[8 + 0j, 0 + 8j]],
    ], dtype=np.complex64)

    spec_x = np.fft.fftshift(
        np.fft.fft(voltages[:, 0, 0].reshape(4, 2), axis=1) / 2**0.5,
        axes=1,
    )
    spec_y = np.fft.fftshift(
        np.fft.fft(voltages[:, 0, 1].reshape(4, 2), axis=1) / 2**0.5,
        axes=1,
    )
    fine_power = np.abs(spec_x) ** 2 + np.abs(spec_y) ** 2
    expected = fine_power.reshape(2, 2, 2).sum(axis=1)

    reduced = _channelize_block(voltages,
                                fftlength=2,
                                integration_factor=2,
                                pol_mode=1,
                                backend="numpy")

    assert reduced.shape == (2, 1, 2)
    assert_allclose(reduced[:, 0, :], expected)


def test_channelize_block_selected_fine_matches_full_slice():
    rng = np.random.default_rng(22)
    voltages = (
        rng.standard_normal((16, 3, 2))
        + 1j * rng.standard_normal((16, 3, 2))
    ).astype(np.complex64)

    full = _channelize_block(voltages,
                             fftlength=4,
                             integration_factor=2,
                             pol_mode=1,
                             backend="numpy",
                             fine_method="full")
    selected_indices = np.arange(3, 9)
    selected = _channelize_block(voltages,
                                 fftlength=4,
                                 integration_factor=2,
                                 pol_mode=1,
                                 backend="numpy",
                                 channel_indices=selected_indices,
                                 fine_method="selected")

    assert_allclose(selected[:, 0, :], full[:, 0, selected_indices], rtol=1e-6, atol=1e-6)


def test_channelize_block_selected_fine_full_stokes_matches_full_slice():
    rng = np.random.default_rng(23)
    voltages = (
        rng.standard_normal((16, 3, 2))
        + 1j * rng.standard_normal((16, 3, 2))
    ).astype(np.complex64)

    full = _channelize_block(voltages,
                             fftlength=4,
                             integration_factor=2,
                             pol_mode=-4,
                             backend="numpy",
                             fine_method="full")
    selected_indices = np.arange(2, 10)
    selected = _channelize_block(voltages,
                                 fftlength=4,
                                 integration_factor=2,
                                 pol_mode=-4,
                                 backend="numpy",
                                 channel_indices=selected_indices,
                                 fine_method="selected")

    assert_allclose(selected, full[:, :, selected_indices], rtol=1e-6, atol=1e-6)


def test_channelize_block_rejects_full_pol_single_pol_input():
    voltages = np.ones((4, 1, 1), dtype=np.complex64)

    with pytest.raises(ValueError, match="dual-polarization"):
        _channelize_block(voltages,
                          fftlength=2,
                          integration_factor=1,
                          pol_mode=4,
                          backend="numpy")


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


def test_reduce_raw_channel_subset_matches_full_coarse_slice(tmp_path):
    backend = _make_backend(num_pols=2, num_bits=8, num_chans=4, fftlength=8, tchans_per_block=8)
    raw_stem = tmp_path / "subset"

    backend.record(output_file_stem=raw_stem,
                   num_blocks=1,
                   length_mode="num_blocks",
                   verbose=False)

    full_spec = stg.voltage.RawReductionSpec(fftlength=8,
                                             integration_factor=1,
                                             pol_mode=1,
                                             output_format="fil")
    subset_spec = stg.voltage.RawReductionSpec(fftlength=8,
                                               integration_factor=1,
                                               pol_mode=1,
                                               output_format="fil",
                                               start_chan=1,
                                               num_chans=2)

    full_frame = stg.voltage.reduce_raw_to_frame(raw_stem, full_spec)
    subset_frame = stg.voltage.reduce_raw_to_frame(raw_stem, subset_spec)

    assert subset_frame.data.shape == (full_frame.tchans, 16)
    assert_allclose(subset_frame.data, full_frame.data[:, 8:24])
    assert subset_frame.df == pytest.approx(full_frame.df)
    assert subset_frame.dt == pytest.approx(full_frame.dt)
    assert subset_frame.fch1 == pytest.approx(full_frame.fch1 + full_frame.df * 8)


def test_reduce_raw_channel_subset_writes_consistent_filterbank_header(tmp_path):
    backend = _make_backend(num_pols=2, num_bits=8, num_chans=4, fftlength=8, tchans_per_block=8)
    raw_stem = tmp_path / "subset_header"
    output_path = tmp_path / "subset_header.fil"

    backend.record(output_file_stem=raw_stem,
                   num_blocks=1,
                   length_mode="num_blocks",
                   verbose=False)

    spec = stg.voltage.RawReductionSpec(fftlength=8,
                                        integration_factor=2,
                                        pol_mode=1,
                                        output_format="fil",
                                        start_chan=2,
                                        num_chans=1)
    stg.voltage.reduce_raw(raw_stem, output_path, spec, overwrite=True)

    wf = Waterfall(str(output_path))
    expected_df_mhz = abs(backend.chan_bw) / spec.fftlength * 1e-6
    expected_fch1_mhz = (backend.fch1 + spec.start_chan * backend.chan_bw - backend.chan_bw / 2) * 1e-6

    assert wf.header["nifs"] == 1
    assert wf.header["nchans"] == spec.fftlength
    assert wf.header["foff"] == pytest.approx(expected_df_mhz)
    assert wf.header["fch1"] == pytest.approx(expected_fch1_mhz)
    assert wf.header["tsamp"] == pytest.approx(backend.tbin * spec.fftlength * spec.integration_factor)
    assert wf.data.shape == (4, 1, spec.fftlength)


def test_reduce_raw_frequency_range_matches_full_fine_slice(tmp_path):
    backend = _make_backend(num_pols=2, num_bits=8, num_chans=4, fftlength=8, tchans_per_block=8)
    raw_stem = tmp_path / "fine_subset"

    backend.record(output_file_stem=raw_stem,
                   num_blocks=1,
                   length_mode="num_blocks",
                   verbose=False)

    full_spec = stg.voltage.RawReductionSpec(fftlength=8,
                                             integration_factor=1,
                                             pol_mode=1,
                                             output_format="fil")
    full_frame = stg.voltage.reduce_raw_to_frame(raw_stem, full_spec)
    start = 5
    stop = 19
    f0 = full_frame.fch1 + full_frame.df * start
    f1 = full_frame.fch1 + full_frame.df * (stop - 1)
    subset_spec = stg.voltage.RawReductionSpec(fftlength=8,
                                               integration_factor=1,
                                               pol_mode=1,
                                               output_format="fil",
                                               frequency_range=(f0, f1),
                                               fine_method="selected")
    subset_frame = stg.voltage.reduce_raw_to_frame(raw_stem, subset_spec)

    assert subset_frame.data.shape == (full_frame.tchans, stop - start)
    assert subset_frame.fch1 == pytest.approx(f0)
    assert subset_frame.df == pytest.approx(full_frame.df)
    assert_allclose(subset_frame.data, full_frame.data[:, start:stop], rtol=1e-6, atol=1e-6)


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


def test_direct_spectrogram_matches_raw_roundtrip_total_power(tmp_path):
    raw_backend = _make_backend(num_pols=2, num_bits=8, num_chans=4, fftlength=8, tchans_per_block=8)
    direct_backend = _make_backend(num_pols=2, num_bits=8, num_chans=4, fftlength=8, tchans_per_block=8)
    raw_stem = tmp_path / "direct_total"

    raw_backend.record(output_file_stem=raw_stem,
                       num_blocks=1,
                       length_mode="num_blocks",
                       verbose=False)

    raw_spec = stg.voltage.RawReductionSpec(fftlength=8,
                                            integration_factor=1,
                                            pol_mode=1,
                                            output_format="fil")
    direct_spec = stg.voltage.VoltageSpectrogramSpec(fftlength=8,
                                                     integration_factor=1,
                                                     pol_mode=1,
                                                     coarse_method="full",
                                                     fine_method="full")
    raw_frame = stg.voltage.reduce_raw_to_frame(raw_stem, raw_spec)
    direct_frame = direct_backend.to_spectrogram(direct_spec,
                                                num_blocks=1,
                                                length_mode="num_blocks",
                                                verbose=False).to_frame()

    assert direct_frame.data.shape == raw_frame.data.shape
    assert_allclose(direct_frame.data, raw_frame.data)


def test_direct_spectrogram_frequency_range_matches_full_direct():
    full_backend = _make_backend(num_pols=2, num_bits=8, num_chans=4, fftlength=8, tchans_per_block=8)
    subset_backend = _make_backend(num_pols=2, num_bits=8, num_chans=4, fftlength=8, tchans_per_block=8)

    full_spec = stg.voltage.VoltageSpectrogramSpec(fftlength=8,
                                                   integration_factor=1,
                                                   pol_mode=1,
                                                   coarse_method="full",
                                                   fine_method="full")
    full_result = full_backend.to_spectrogram(full_spec,
                                             num_blocks=1,
                                             length_mode="num_blocks",
                                             verbose=False)
    full_frame = full_result.to_frame()
    start = 4
    stop = 12
    f0 = full_frame.fch1 + full_frame.df * start
    f1 = full_frame.fch1 + full_frame.df * (stop - 1)
    subset_spec = stg.voltage.VoltageSpectrogramSpec(fftlength=8,
                                                     integration_factor=1,
                                                     pol_mode=1,
                                                     frequency_range=(f0, f1),
                                                     coarse_method="full",
                                                     fine_method="selected")
    subset_frame = subset_backend.to_spectrogram(subset_spec,
                                                num_blocks=1,
                                                length_mode="num_blocks",
                                                verbose=False).to_frame()

    assert subset_frame.data.shape == (full_frame.tchans, stop - start)
    assert subset_frame.fch1 == pytest.approx(f0)
    assert subset_frame.df == pytest.approx(full_frame.df)
    assert_allclose(subset_frame.data, full_frame.data[:, start:stop], rtol=1e-6, atol=1e-6)


def test_direct_spectrogram_matches_raw_roundtrip_full_stokes_4bit(tmp_path):
    raw_backend = _make_backend(num_pols=2, num_bits=4, num_chans=4, fftlength=8, tchans_per_block=8)
    direct_backend = _make_backend(num_pols=2, num_bits=4, num_chans=4, fftlength=8, tchans_per_block=8)
    raw_stem = tmp_path / "direct_stokes_4bit"

    raw_backend.record(output_file_stem=raw_stem,
                       num_blocks=1,
                       length_mode="num_blocks",
                       verbose=False)

    raw_spec = stg.voltage.RawReductionSpec(fftlength=8,
                                            integration_factor=1,
                                            pol_mode=-4,
                                            output_format="fil")
    direct_spec = stg.voltage.VoltageSpectrogramSpec(fftlength=8,
                                                     integration_factor=1,
                                                     pol_mode=-4,
                                                     coarse_method="full",
                                                     fine_method="full")
    raw_chunks = []
    for chunk, _, _ in _reduce_chunks(raw_stem, raw_spec):
        raw_chunks.append(chunk)
    raw_data = np.concatenate(raw_chunks, axis=0)
    direct_data = direct_backend.to_spectrogram(direct_spec,
                                               num_blocks=1,
                                               length_mode="num_blocks",
                                               verbose=False).data

    assert direct_data.shape == raw_data.shape
    assert_allclose(direct_data, raw_data)


def test_direct_spectrogram_from_data_shape(tmp_path):
    base_backend = _make_backend(num_pols=2, num_bits=8, num_chans=4, fftlength=8, tchans_per_block=8)
    raw_stem = tmp_path / "direct_from_data"
    base_backend.record(output_file_stem=raw_stem,
                        num_blocks=1,
                        length_mode="num_blocks",
                        verbose=False)

    raw_params = stg.voltage.get_raw_params(input_file_stem=raw_stem,
                                            start_chan=0)
    antenna = stg.voltage.Antenna(sample_rate=base_backend.sample_rate,
                                  seed=123,
                                  **raw_params)
    read_backend = stg.voltage.RawVoltageBackend.from_data(input_file_stem=raw_stem,
                                                           antenna_source=antenna,
                                                           start_chan=0,
                                                           num_subblocks=4)
    spec = stg.voltage.VoltageSpectrogramSpec(fftlength=8,
                                              integration_factor=1,
                                              pol_mode=1,
                                              coarse_method="full",
                                              fine_method="full")
    result = read_backend.to_spectrogram(spec,
                                         num_blocks=1,
                                         length_mode="num_blocks",
                                         verbose=False)

    assert result.data.shape == (8, 1, 32)


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
