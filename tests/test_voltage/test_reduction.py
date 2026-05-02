import importlib.util
from types import SimpleNamespace

import numpy as np
from numpy.testing import assert_allclose
import pytest
from astropy import units as u
from astropy.time import Time
from blimpy import Waterfall

import setigen as stg
import setigen.voltage._reduction.channelize as channelize_module
import setigen.voltage.reduction as reduction_module
import setigen.voltage.spectrogram as spectrogram_module
from setigen.voltage._reduction.metadata import _ReductionMetadata, _build_reduction_metadata
from setigen.voltage._reduction.channelize import _channelize_block
from setigen.voltage._reduction.decoder import _decode_raw_block
from setigen.voltage.reduction import _reduce_chunks
from setigen.voltage.spectrogram import _to_host_array


def _require_cupy_device():
    if importlib.util.find_spec("cupy") is None:
        pytest.skip("CuPy is not installed.")
    cupy = pytest.importorskip("cupy")
    try:
        device_count = cupy.cuda.runtime.getDeviceCount()
    except Exception as exc:
        pytest.skip(f"CuPy CUDA runtime is unavailable: {exc}")
    if device_count < 1:
        pytest.skip("No CUDA devices available.")
    return cupy


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


def test_channelize_block_auto_and_full_slice_channel_selection():
    rng = np.random.default_rng(24)
    voltages = (
        rng.standard_normal((16, 3, 2))
        + 1j * rng.standard_normal((16, 3, 2))
    ).astype(np.complex64)
    selected_indices = np.arange(2, 8)

    full_total = _channelize_block(voltages,
                                   fftlength=4,
                                   integration_factor=2,
                                   pol_mode=1,
                                   backend="numpy",
                                   fine_method="full")
    auto_selected = _channelize_block(voltages,
                                      fftlength=4,
                                      integration_factor=2,
                                      pol_mode=1,
                                      backend="numpy",
                                      channel_indices=selected_indices)
    full_selected = _channelize_block(voltages,
                                      fftlength=4,
                                      integration_factor=2,
                                      pol_mode=1,
                                      backend="numpy",
                                      channel_indices=selected_indices,
                                      fine_method="full")
    full_stokes = _channelize_block(voltages,
                                    fftlength=4,
                                    integration_factor=2,
                                    pol_mode=-4,
                                    backend="numpy",
                                    fine_method="full")
    full_stokes_selected = _channelize_block(voltages,
                                             fftlength=4,
                                             integration_factor=2,
                                             pol_mode=-4,
                                             backend="numpy",
                                             channel_indices=selected_indices,
                                             fine_method="full")

    assert_allclose(auto_selected[:, 0, :],
                    full_total[:, 0, selected_indices],
                    rtol=1e-6,
                    atol=1e-6)
    assert_allclose(full_selected[:, 0, :],
                    full_total[:, 0, selected_indices],
                    rtol=1e-6,
                    atol=1e-6)
    assert_allclose(full_stokes_selected,
                    full_stokes[:, :, selected_indices],
                    rtol=1e-6,
                    atol=1e-6)


def test_channelize_block_rejects_full_pol_single_pol_input():
    voltages = np.ones((4, 1, 1), dtype=np.complex64)

    with pytest.raises(ValueError, match="dual-polarization"):
        _channelize_block(voltages,
                          fftlength=2,
                          integration_factor=1,
                          pol_mode=4,
                          backend="numpy")


def test_channelize_block_edge_cases():
    voltages = np.ones((3, 1, 2), dtype=np.complex64)

    with pytest.raises(ValueError, match="fine_method must be"):
        _channelize_block(voltages,
                          fftlength=2,
                          integration_factor=1,
                          pol_mode=1,
                          backend="numpy",
                          fine_method="bad")

    with pytest.raises(ValueError, match="out of bounds"):
        _channelize_block(voltages,
                          fftlength=2,
                          integration_factor=1,
                          pol_mode=1,
                          backend="numpy",
                          channel_indices=[2],
                          fine_method="selected")

    with pytest.raises(ValueError, match="Unsupported reduction backend"):
        _channelize_block(voltages,
                          fftlength=2,
                          integration_factor=1,
                          pol_mode=1,
                          backend="bad")

    with pytest.raises(ValueError, match="Unsupported polarization mode"):
        _channelize_block(voltages,
                          fftlength=2,
                          integration_factor=1,
                          pol_mode=2,
                          backend="numpy")

    assert _channelize_block(voltages[:1],
                             fftlength=2,
                             integration_factor=1,
                             pol_mode=1,
                             backend="numpy") is None
    assert _channelize_block(voltages[:1],
                             fftlength=2,
                             integration_factor=1,
                             pol_mode=1,
                             backend="numpy",
                             channel_indices=[0],
                             fine_method="selected") is None
    assert _channelize_block(voltages,
                             fftlength=2,
                             integration_factor=2,
                             pol_mode=1,
                             backend="numpy") is None
    assert _channelize_block(voltages,
                             fftlength=2,
                             integration_factor=1,
                             pol_mode=1,
                             backend="numpy",
                             channel_indices=[],
                             fine_method="selected") is None
    assert _channelize_block(np.ones((2, 1, 2), dtype=np.complex64),
                             fftlength=2,
                             integration_factor=2,
                             pol_mode=-4,
                             backend="numpy") is None


def test_channelize_block_converts_non_numpy_backend_to_host(monkeypatch):
    class FakeArrayModule:
        __name__ = "fake"
        fft = np.fft
        newaxis = np.newaxis
        pi = np.pi

        @staticmethod
        def asarray(array):
            return np.asarray(array)

        @staticmethod
        def asnumpy(array):
            return np.asarray(array)

        @staticmethod
        def abs(array):
            return np.abs(array)

    monkeypatch.setattr(channelize_module,
                        "_get_array_module",
                        lambda backend: FakeArrayModule)
    voltages = np.ones((4, 1, 1), dtype=np.complex64)

    reduced = _channelize_block(voltages,
                                fftlength=2,
                                integration_factor=1,
                                pol_mode=1,
                                backend="fake")

    assert isinstance(reduced, np.ndarray)
    assert reduced.dtype == np.float32


def test_reduction_specs_reject_invalid_new_options():
    base_kwargs = {
        "fftlength": 8,
        "integration_factor": 1,
        "pol_mode": 1,
        "output_format": "fil",
    }

    invalid_kwargs = [
        {"fftlength": 0, "match": "fftlength"},
        {"integration_factor": 0, "match": "integration_factor"},
        {"pol_mode": 2, "match": "polarization"},
        {"output_format": "bad", "match": "output format"},
        {"backend": "bad", "match": "backend"},
        {"accuracy": "bad", "match": "accuracy"},
        {"accuracy": "approx_zoom", "match": "accuracy='exact'"},
        {"coarse_method": "bad", "match": "coarse method"},
        {"fine_method": "bad", "match": "fine method"},
        {"start_chan": -1, "match": "start_chan"},
        {"num_chans": 0, "match": "num_chans"},
        {"frequency_range": (1.0, 2.0), "start_chan": 0, "match": "mutually exclusive"},
        {"frequency_range": (1.0, 2.0, 3.0), "match": "exactly two"},
    ]
    for params in invalid_kwargs:
        kwargs = dict(params)
        match = kwargs.pop("match")
        with pytest.raises(ValueError, match=match):
            stg.voltage.RawReductionSpec(**{**base_kwargs, **kwargs})

    spectrogram_base = {
        "fftlength": 8,
        "integration_factor": 1,
        "pol_mode": 1,
    }
    for params in invalid_kwargs:
        if "output_format" in params:
            continue
        kwargs = dict(params)
        match = kwargs.pop("match")
        with pytest.raises(ValueError, match=match):
            stg.voltage.VoltageSpectrogramSpec(**{**spectrogram_base, **kwargs})


def test_build_reduction_metadata_frequency_range_and_bounds():
    input_spec = SimpleNamespace(
        num_chans=4,
        chan_bw=8.0,
        tbin=0.5,
        fch1=100.0,
        ascending=True,
    )
    spec = stg.voltage.RawReductionSpec(fftlength=4,
                                        integration_factor=2,
                                        pol_mode=1,
                                        output_format="fil",
                                        frequency_range=(99.0, 105.0))

    metadata = _build_reduction_metadata(input_spec, spec)

    assert metadata.channel_start == 2
    assert metadata.channel_stop == 5
    assert metadata.total_fchans == 3
    assert metadata.fch1_hz == pytest.approx(100.0)

    out_of_range = stg.voltage.RawReductionSpec(fftlength=4,
                                                integration_factor=2,
                                                pol_mode=1,
                                                output_format="fil",
                                                frequency_range=(1.0, 2.0))
    with pytest.raises(ValueError, match="does not overlap"):
        _build_reduction_metadata(input_spec, out_of_range)

    coarse_out_of_bounds = stg.voltage.RawReductionSpec(fftlength=4,
                                                        integration_factor=2,
                                                        pol_mode=1,
                                                        output_format="fil",
                                                        start_chan=4,
                                                        num_chans=1)
    with pytest.raises(ValueError, match="out of bounds"):
        _build_reduction_metadata(input_spec, coarse_out_of_bounds)

    conflicting = SimpleNamespace(start_chan=0,
                                  num_chans=None,
                                  frequency_range=(1.0, 2.0),
                                  pol_mode=1,
                                  fftlength=4,
                                  integration_factor=2)
    with pytest.raises(ValueError, match="mutually exclusive"):
        _build_reduction_metadata(input_spec, conflicting)


def test_voltage_spectrogram_result_error_paths(tmp_path):
    metadata = _ReductionMetadata(start_chan=0,
                                  num_chans=1,
                                  nifs=4,
                                  df_hz=1.0,
                                  dt_s=1.0,
                                  fch1_hz=100.0,
                                  ascending=True,
                                  total_fchans=4)
    result = stg.voltage.VoltageSpectrogramResult(
        data=np.zeros((1, 4, 4), dtype=np.float32),
        metadata=metadata,
        spec=stg.voltage.VoltageSpectrogramSpec(fftlength=4,
                                                integration_factor=1,
                                                pol_mode=-4),
        input_spec=SimpleNamespace(),
    )

    with pytest.raises(ValueError, match="to_frame only supports"):
        result.to_frame()
    with pytest.raises(ValueError, match="output_format must be supplied"):
        result.write(tmp_path / "spectrogram.txt")


def test_voltage_spectrogram_result_write_infers_supported_suffixes(monkeypatch, tmp_path):
    class DummyWriter:
        def __init__(self):
            self.appended = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def append(self, data):
            self.appended.append(data)

    writers = []

    def create_writer(*args, **kwargs):
        writer = DummyWriter()
        writers.append((writer, kwargs["output_format"]))
        return writer

    monkeypatch.setattr(spectrogram_module,
                        "_build_filterbank_header",
                        lambda *args, **kwargs: {})
    monkeypatch.setattr(spectrogram_module, "_create_writer", create_writer)
    metadata = _ReductionMetadata(start_chan=0,
                                  num_chans=1,
                                  nifs=1,
                                  df_hz=1.0,
                                  dt_s=1.0,
                                  fch1_hz=100.0,
                                  ascending=True,
                                  total_fchans=4)
    result = stg.voltage.VoltageSpectrogramResult(
        data=np.zeros((1, 1, 4), dtype=np.float32),
        metadata=metadata,
        spec=stg.voltage.VoltageSpectrogramSpec(fftlength=4,
                                                integration_factor=1,
                                                pol_mode=1),
        input_spec=SimpleNamespace(),
    )

    assert result.write(tmp_path / "spectrogram.fil") == tmp_path / "spectrogram.fil"
    assert result.write(tmp_path / "spectrogram.h5") == tmp_path / "spectrogram.h5"
    assert [output_format for _, output_format in writers] == ["fil", "h5"]
    assert all(writer.appended for writer, _ in writers)


def test_raw_reduction_no_output_error_paths(monkeypatch, tmp_path):
    spec = stg.voltage.RawReductionSpec(fftlength=4,
                                        integration_factor=1,
                                        pol_mode=1,
                                        output_format="fil")
    monkeypatch.setattr(reduction_module,
                        "_reduce_chunks",
                        lambda input_path, spec, max_blocks=None: iter(()))

    with pytest.raises(ValueError, match="No spectra were produced"):
        reduction_module.reduce_raw(tmp_path / "missing.raw",
                                    tmp_path / "missing.fil",
                                    spec)
    with pytest.raises(ValueError, match="No frame data"):
        reduction_module.reduce_raw_to_frame(tmp_path / "missing.raw", spec)

    full_stokes = stg.voltage.RawReductionSpec(fftlength=4,
                                               integration_factor=1,
                                               pol_mode=-4,
                                               output_format="fil")
    with pytest.raises(ValueError, match="only supports total-power"):
        reduction_module.reduce_raw_to_frame(tmp_path / "missing.raw", full_stokes)


def test_generate_voltage_spectrogram_raises_when_no_chunks(monkeypatch):
    backend = SimpleNamespace(
        num_chans=1,
        num_pols=2,
        num_bits=8,
        chan_bw=8.0,
        tbin=0.5,
        fch1=100.0,
        start_chan=0,
        ascending=True,
        input_num_blocks=None,
        input_file_stem=None,
        time_per_block=1.0,
        num_branches=1,
        get_num_blocks=lambda length: 1,
    )
    metadata = _ReductionMetadata(start_chan=0,
                                  num_chans=1,
                                  nifs=1,
                                  df_hz=1.0,
                                  dt_s=1.0,
                                  fch1_hz=100.0,
                                  ascending=True,
                                  total_fchans=4)

    monkeypatch.setattr(spectrogram_module,
                        "_build_reduction_metadata",
                        lambda input_spec, spec: metadata)
    monkeypatch.setattr(spectrogram_module._RecordConfig,
                        "from_values",
                        lambda **kwargs: SimpleNamespace(length=1))
    monkeypatch.setattr(spectrogram_module, "_resolve_num_blocks", lambda *args, **kwargs: 1)
    monkeypatch.setattr(spectrogram_module, "_reset_recording_state", lambda backend: None)
    monkeypatch.setattr(spectrogram_module, "_get_num_output_files", lambda backend, xp: 1)
    monkeypatch.setattr(spectrogram_module,
                        "_get_blocks_to_write",
                        lambda backend, file_index, num_files: 1)
    monkeypatch.setattr(spectrogram_module,
                        "_collect_coarse_voltage_block",
                        lambda *args, **kwargs: np.ones((4, 1, 2), dtype=np.complex64))
    monkeypatch.setattr(spectrogram_module, "_channelize_block", lambda *args, **kwargs: None)

    with pytest.raises(ValueError, match="No spectra were produced"):
        spectrogram_module.generate_voltage_spectrogram(
            backend,
            stg.voltage.VoltageSpectrogramSpec(fftlength=4, integration_factor=1),
            num_blocks=1,
            length_mode="num_blocks",
            verbose=False,
            xp=np,
        )


def test_collect_coarse_voltage_block_requires_requantize_for_raw_input(monkeypatch):
    backend = SimpleNamespace(num_chans=1,
                              num_antennas=1,
                              num_pols=2,
                              input_file_stem="input")
    metadata = SimpleNamespace(num_chans=1, start_chan=0)
    spec = stg.voltage.VoltageSpectrogramSpec(fftlength=4,
                                              integration_factor=1)
    monkeypatch.setattr(spectrogram_module,
                        "_plan_subblocks",
                        lambda backend, obsnchan: SimpleNamespace(
                            num_subblocks=1,
                            total_time_samples=1,
                            bytes_per_subblock=1,
                        ))

    with pytest.raises(ValueError, match="requantize=True"):
        spectrogram_module._collect_coarse_voltage_block(
            backend,
            metadata=metadata,
            spec=spec,
            digitize=True,
            requantize=False,
            verbose=False,
            xp=np,
        )


def test_to_host_array_uses_backend_asnumpy():
    class FakeArrayModule:
        @staticmethod
        def asnumpy(array):
            return np.asarray(array) + 1

    assert np.array_equal(_to_host_array(np.array([1]), xp=FakeArrayModule), np.array([2]))
    original = np.array([1])
    assert _to_host_array(original, xp=np) is original


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
    assert frame.header is not None
    assert frame.source_name == frame.header["source_name"]
    assert frame.t_start == pytest.approx(Time(frame.header["tstart"], format="mjd").unix)


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
    assert direct_frame.header is not None
    assert direct_frame.source_name == direct_frame.header["source_name"]
    assert direct_frame.t_start == pytest.approx(Time(direct_frame.header["tstart"], format="mjd").unix)


def test_direct_spectrogram_cupy_backend_smoke():
    _require_cupy_device()
    try:
        stg.voltage.set_backend("cupy")
        backend = _make_backend(num_pols=2,
                                num_bits=8,
                                num_chans=4,
                                fftlength=8,
                                tchans_per_block=8)
        spec = stg.voltage.VoltageSpectrogramSpec(fftlength=8,
                                                  integration_factor=1,
                                                  pol_mode=1,
                                                  backend="cupy",
                                                  coarse_method="full",
                                                  fine_method="full")

        result = backend.to_spectrogram(spec,
                                        num_blocks=1,
                                        length_mode="num_blocks",
                                        verbose=False)

        assert result.data.shape == (8, 1, 32)
        assert np.isfinite(result.data).all()
    finally:
        stg.voltage.set_backend("numpy")


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
