import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_allclose
import pytest

from astropy import units as u
import setigen as stg
from setigen._frame import file_mutation
from setigen._frame.context import _copy_frame_context, _finalize_derived_frame
from setigen._frame.io import _close_waterfall_handles, _has_live_h5_handle
from setigen._frame.signal import _get_profile_support_half_width
from setigen._spectrogram import copy_spectrogram, open_spectrogram
from setigen._spectrogram.fil import _HEADER_KEYWORD_TYPES, _dtype_from_nbits, _read_keyword


def _write_frame(frame, path):
    if path.suffix == ".h5":
        frame.save_hdf5(path)
    else:
        frame.save_fil(path)


def _signal_kwargs(frame):
    return dict(
        path=stg.constant_path(frame.get_frequency(frame.fchans // 2), drift_rate=0),
        t_profile=stg.constant_t_profile(3),
        f_profile=stg.box_f_profile(width=4 * frame.df),
        auto_bounding=True,
    )


def _contract_signal_kwargs(frame, case):
    if case == "drifted_integrated_gaussian":
        return dict(
            path=stg.constant_path(
                frame.get_frequency(frame.fchans // 2 - 6),
                drift_rate=0.35 * frame.unit_drift_rate,
            ),
            t_profile=stg.sine_t_profile(
                period=4 * frame.dt,
                amplitude=0.4,
                level=2.0,
            ),
            f_profile=stg.gaussian_f_profile(width=5 * frame.df),
            bp_profile=stg.constant_bp_profile(level=1),
            integrate_path=True,
            integrate_t_profile=True,
            integrate_f_profile=True,
            t_subsamples=4,
            f_subsamples=4,
            auto_bounding=True,
            truncate_below=1e-3,
        )
    if case == "doppler_smeared_sinc2":
        return dict(
            path=stg.constant_path(
                frame.get_frequency(frame.fchans // 2),
                drift_rate=1.5 * frame.unit_drift_rate,
            ),
            t_profile=stg.constant_t_profile(level=1.5),
            f_profile=stg.sinc2_f_profile(width=4 * frame.df),
            bp_profile=stg.constant_bp_profile(level=1),
            doppler_smearing=True,
            smearing_subsamples=5,
            auto_bounding=True,
        )
    raise ValueError(f"Unknown signal contract case {case}")


@pytest.mark.parametrize("suffix", [".h5", ".fil"])
def test_file_backed_open_copy_add_signal_matches_eager(tmp_path, suffix):
    source_path = tmp_path / f"source{suffix}"
    output_path = tmp_path / f"injected{suffix}"

    source_frame = stg.Frame(tchans=8,
                             fchans=64,
                             df=2,
                             dt=1,
                             fch1=1062,
                             ascending=False,
                             source_name="Target")
    source_frame.data[:] = np.arange(np.prod(source_frame.shape),
                                     dtype=np.float32).reshape(source_frame.shape)
    _write_frame(source_frame, source_path)

    eager = stg.Frame(waterfall=source_path)
    kwargs = _signal_kwargs(eager)
    eager.add_signal(**kwargs)

    with stg.Frame.open_copy(source_path,
                             output_path,
                             max_chunk_bytes=256) as frame:
        result = frame.add_signal(**kwargs)
        assert frame.is_file_backed
        assert result.time_chunks > 1

        region = frame.read_frame(f_index_range=(24, 36),
                                  t_index_range=(0, frame.tchans))
        assert_allclose(region.data, eager.data[:, 24:36])

        artist = frame.plot(f_index_range=(24, 36),
                            t_index_range=(0, frame.tchans),
                            db=False,
                            colorbar=False)
        assert artist is not None
        plt.close()

    loaded_output = stg.Frame(waterfall=output_path)
    assert_allclose(loaded_output.data, eager.data)

    loaded_source = stg.Frame(waterfall=source_path)
    assert_allclose(loaded_source.data, source_frame.data)


@pytest.mark.parametrize("suffix", [".h5", ".fil"])
@pytest.mark.parametrize("ascending", [False, True])
@pytest.mark.parametrize("case", ["drifted_integrated_gaussian", "doppler_smeared_sinc2"])
def test_file_backed_add_signal_contract_matches_eager_for_common_options(
    tmp_path,
    suffix,
    ascending,
    case,
):
    source_path = tmp_path / f"source_{ascending}_{case}{suffix}"
    output_path = tmp_path / f"injected_{ascending}_{case}{suffix}"
    source_frame = stg.Frame(tchans=8,
                             fchans=96,
                             df=1,
                             dt=1,
                             fch1=6e9 + (0 if ascending else 95),
                             ascending=ascending,
                             source_name="Target")
    source_frame.data[:] = np.arange(np.prod(source_frame.shape),
                                     dtype=np.float32).reshape(source_frame.shape) / 100
    _write_frame(source_frame, source_path)

    eager = stg.Frame(waterfall=source_path)
    kwargs = _contract_signal_kwargs(eager, case)
    eager.add_signal(**kwargs)

    with stg.Frame.open_copy(source_path,
                             output_path,
                             max_chunk_bytes=256) as backed:
        result = backed.add_signal(**kwargs)
        assert result.time_chunks > 1

    loaded_output = stg.Frame(waterfall=output_path)
    loaded_source = stg.Frame(waterfall=source_path)
    assert_allclose(loaded_output.data, eager.data, rtol=1e-6, atol=1e-6)
    assert_allclose(loaded_source.data, source_frame.data)


@pytest.mark.parametrize("suffix", [".h5", ".fil"])
@pytest.mark.parametrize("ascending", [False, True])
@pytest.mark.parametrize("channel_index", [0, 1, 37, 94, 95])
def test_file_backed_add_signal_preserves_exact_channel_index(
    tmp_path,
    suffix,
    ascending,
    channel_index,
):
    source_path = tmp_path / f"source_{ascending}_{channel_index}{suffix}"
    output_path = tmp_path / f"injected_{ascending}_{channel_index}{suffix}"
    source_frame = stg.Frame(tchans=5,
                             fchans=96,
                             df=1,
                             dt=1,
                             fch1=6e9 + (0 if ascending else 95),
                             ascending=ascending,
                             source_name="Target")
    _write_frame(source_frame, source_path)

    eager = stg.Frame(waterfall=source_path)
    kwargs = dict(
        path=stg.constant_path(eager.get_frequency(channel_index), drift_rate=0),
        t_profile=stg.constant_t_profile(level=7),
        f_profile=stg.box_f_profile(width=eager.df),
        bp_profile=stg.constant_bp_profile(level=1),
        auto_bounding=True,
    )
    eager.add_signal(**kwargs)

    with stg.Frame.open_copy(source_path,
                             output_path,
                             max_chunk_bytes=64) as backed:
        result = backed.add_signal(**kwargs)
        assert result.time_chunks > 1

    loaded_output = stg.Frame(waterfall=output_path)
    loaded_source = stg.Frame(waterfall=source_path)
    expected_delta = np.zeros(source_frame.shape)
    expected_delta[:, channel_index] = 7
    assert_allclose(loaded_output.data, eager.data, rtol=1e-6, atol=1e-6)
    assert_allclose(loaded_output.data - loaded_source.data, expected_delta)
    assert_allclose(loaded_source.data, source_frame.data)


@pytest.mark.parametrize("suffix", [".h5", ".fil"])
def test_file_backed_read_only_frames_do_not_mutate(tmp_path, suffix):
    source_path = tmp_path / f"source{suffix}"
    frame = stg.Frame(tchans=4, fchans=16, df=1, dt=1, fch1=1015)
    _write_frame(frame, source_path)

    with stg.Frame.open(source_path) as backed:
        assert backed.is_file_backed
        with pytest.raises(OSError, match="read-only"):
            backed.add_signal(**_signal_kwargs(backed))


def test_file_backed_direct_write_requires_explicit_guard(tmp_path):
    source_path = tmp_path / "source.h5"
    stg.Frame(tchans=4, fchans=16).save_hdf5(source_path)

    with pytest.raises(ValueError, match="allow_inplace"):
        stg.Frame.open(source_path, mode="r+")


@pytest.mark.parametrize("suffix", [".h5", ".fil"])
def test_file_backed_noise_stats_use_local_context(tmp_path, suffix):
    source_path = tmp_path / f"source{suffix}"
    frame = stg.Frame(tchans=8, fchans=64, df=1, dt=1, fch1=1063)
    frame.data[:] = 1000
    local_pattern = np.array([9, 11] * 8, dtype=np.float32)
    frame.data[:, 24:40] = local_pattern
    _write_frame(frame, source_path)

    with stg.Frame.open(source_path) as backed:
        with pytest.raises(ValueError, match="explicit path"):
            backed.estimate_noise_stats()

        stats = backed.estimate_noise_stats(
            bounding_f_range=(backed.get_frequency(30),
                              backed.get_frequency(34)),
            t_index_range=(0, backed.tchans),
            config=stg.NoiseEstimationConfig(context_width=6,
                                             guard_width=1),
        )

    assert stats.mean == pytest.approx(10)
    assert stats.std == pytest.approx(1)
    assert stats.context_bounds == (24, 40)
    assert stats.excluded_bounds == (29, 35)


@pytest.mark.parametrize("suffix", [".h5", ".fil"])
def test_file_backed_spectrum_and_timeseries_match_eager(tmp_path, suffix):
    source_path = tmp_path / f"source{suffix}"
    source_frame = stg.Frame(tchans=8,
                             fchans=32,
                             df=2,
                             dt=1,
                             fch1=1062,
                             ascending=False,
                             source_name="Target")
    source_frame.data[:] = np.arange(np.prod(source_frame.shape),
                                     dtype=np.float32).reshape(source_frame.shape)
    _write_frame(source_frame, source_path)

    eager = stg.Frame(waterfall=source_path)
    eager_spectrum = eager.spectrum(mode="sum",
                                    f_index_range=(4, 20),
                                    t_index_range=(2, 7))
    eager_timeseries = eager.timeseries(mode="mean",
                                        f_index_range=(4, 20),
                                        t_index_range=(2, 7))

    with stg.Frame.open(source_path, max_chunk_bytes=64) as backed:
        backed_spectrum = backed.spectrum(mode="sum",
                                          f_index_range=(4, 20),
                                          t_index_range=(2, 7),
                                          max_chunk_bytes=64)
        backed_timeseries = backed.timeseries(mode="mean",
                                              f_index_range=(4, 20),
                                              t_index_range=(2, 7),
                                              max_chunk_bytes=64)

    assert_allclose(backed_spectrum.data, eager_spectrum.data)
    assert_allclose(backed_timeseries.data, eager_timeseries.data)
    assert backed_spectrum.metadata["derived"]["source_bounds"]["frequency_index_range"] == (4, 20)
    assert backed_timeseries.metadata["derived"]["source_bounds"]["time_index_range"] == (2, 7)


def test_file_backed_frame_data_and_noise_guardrails(tmp_path):
    source_path = tmp_path / "source.h5"
    frame = stg.Frame(tchans=4, fchans=8, df=2, dt=1, fch1=6e9)
    frame.data[:] = np.arange(np.prod(frame.shape)).reshape(frame.shape)
    frame.save_hdf5(source_path)

    with pytest.raises(ValueError, match="mode='r'"):
        stg.Frame.open(source_path, mode="w")

    with stg.Frame.open_copy(source_path, tmp_path / "copy.h5") as backed:
        assert_allclose(backed.copy().data, frame.data)
        assert_allclose(backed.data, frame.data)

        backed.data = None
        assert backed._data is None
        with pytest.raises(ValueError, match="Data shape"):
            backed.data = np.zeros((1, 1))

        replacement = np.full(frame.shape, 5, dtype=np.float32)
        backed.data = replacement
        assert_allclose(backed.data, replacement)

        backed._update_noise_frame_stats()
        assert backed.noise_stats is None
        with pytest.raises(NotImplementedError, match="add_noise"):
            backed.add_noise(1)
        with pytest.raises(NotImplementedError, match="add_noise_from_obs"):
            backed.add_noise_from_obs()


def test_frame_noise_bounds_and_range_edges():
    frame = stg.Frame(tchans=4, fchans=16, df=2, dt=1, fch1=6e9, ascending=True)
    frame.data[:] = np.arange(np.prod(frame.shape)).reshape(frame.shape)

    stats = frame.estimate_noise_stats(
        bounding_f_range=(frame.get_frequency(-50), frame.get_frequency(-40)),
        config=stg.NoiseEstimationConfig(context_width=2, guard_width=0),
    )
    assert stats.context_bounds == (0, 3)

    stats = frame.estimate_noise_stats(
        path=frame.get_frequency(5),
        config=stg.NoiseEstimationConfig(context_width=2 * frame.df,
                                         guard_width=0,
                                         width_unit="Hz"),
    )
    assert stats.context_bounds == (3, 8)

    with pytest.raises(ValueError, match="requires path and f_profile"):
        frame.estimate_noise_stats(auto_bounding=True)
    with pytest.raises(ValueError, match="empty"):
        frame.estimate_noise_stats(f_index_range=(2, 2))
    with pytest.raises(ValueError, match="guard removed all"):
        frame.estimate_noise_stats(
            bounding_f_range=(frame.get_frequency(5), frame.get_frequency(6)),
            config=stg.NoiseEstimationConfig(context_width=0, guard_width=10),
        )

    sub = frame.read_frame(f_range=(frame.frequency_edges[2] * u.Hz,
                                    frame.frequency_edges[5] * u.Hz),
                           t_range=(1 * u.s, 3 * u.s))
    assert_allclose(sub.data, frame.data[1:3, 2:5])
    assert sub.fch1 == frame.fs[2]

    with pytest.raises(ValueError, match="empty"):
        frame.read_frame(f_index_range=(3, 3))


def test_frame_misc_edge_contracts():
    frame = stg.Frame(tchans=4, fchans=8, df=1, dt=2, fch1=6e9)
    assert frame.get_intensity(10, noise_stats=(0, 2)) == pytest.approx(10)
    assert frame.get_snr(10, noise_stats=(0, 2)) == pytest.approx(10)
    assert frame.get_drift_rate(0, 2, reference="centers") == pytest.approx(1 / 3)
    with pytest.raises(ValueError, match="at least two"):
        stg.Frame(tchans=1, fchans=8).get_drift_rate(0, 1, reference="centers")
    with pytest.raises(ValueError, match="reference"):
        frame.get_drift_rate(0, 1, reference="bad")
    assert_allclose(frame.integrate(), np.zeros(frame.fchans))

    class BadWaterfall:
        def __deepcopy__(self, memo):
            raise RuntimeError("no copy")

    frame.waterfall = BadWaterfall()
    assert frame.copy().waterfall is None


def test_integration_edge_contracts(tmp_path):
    data = np.arange(12, dtype=np.float32).reshape(3, 4)
    array_like = data.tolist()
    assert_allclose(stg.integrate(array_like, axis="frequency", mode="sum"), np.sum(data, axis=1))
    assert_allclose(stg.integrate(array_like, axis=1, mode="mean"), np.mean(data, axis=1))

    with pytest.raises(ValueError, match="Frame-like"):
        stg.integrate(array_like, f_index_range=(0, 1))
    with pytest.raises(TypeError, match="Frame-like"):
        stg.integrate(array_like, as_frame=True)

    frame = stg.Frame(tchans=3, fchans=4, df=1, dt=1, fch1=6e9, data=data)
    with pytest.raises(ValueError, match="empty"):
        stg.integrate(frame, f_index_range=(2, 2))
    assert_allclose(stg.integrate(frame, axis="frequency"), np.mean(data, axis=1))

    source_path = tmp_path / "integrate_source.h5"
    frame.save_hdf5(source_path)
    with stg.Frame.open(source_path) as backed:
        assert_allclose(backed.spectrum(mode="mean").data, np.mean(data, axis=0, keepdims=True))
        with pytest.raises(ValueError, match="max_chunk_bytes"):
            backed.spectrum(max_chunk_bytes=0)


def test_signal_support_metadata_and_callable_paths():
    assert _get_profile_support_half_width(stg.gaussian_f_profile(width=10), None) is None
    assert _get_profile_support_half_width(stg.multiple_gaussian_f_profile(width=10), 1e-3) > 100
    assert _get_profile_support_half_width(stg.lorentzian_f_profile(width=10), 1e-3) > 0
    assert _get_profile_support_half_width(stg.voigt_f_profile(10, 10), 1e-3) is None

    frame = stg.Frame(tchans=4, fchans=64, df=1, dt=1, fch1=6e9 + 32)
    frame.add_signal(path=lambda ts: 6e9,
                     t_profile=1,
                     f_profile=stg.gaussian_f_profile(width=3),
                     auto_bounding=True)
    assert np.max(frame.data) > 0

    frame.zero_data()
    frame.add_signal(path=lambda ts: 6e9,
                     t_profile=1,
                     f_profile=stg.box_f_profile(width=3),
                     integrate_path=True,
                     auto_bounding=True)
    assert np.max(frame.data) > 0


def test_file_mutation_private_edge_helpers(tmp_path):
    frame = stg.Frame(tchans=4, fchans=8, df=1, dt=1, fch1=6e9, ascending=True)
    chunk = file_mutation._ChunkFrameView(
        frame,
        t_start_index=1,
        tchans=2,
        f_start_index=2,
        f_stop_index=6,
        data=np.zeros((2, 4)),
    )
    assert chunk.get_index(6e9 + 3) == 1
    assert file_mutation._slice_time_input(5, t_start=0, t_stop=1, full_tchans=4) == 5
    mismatched = np.arange(3)
    assert file_mutation._slice_time_input(mismatched,
                                           t_start=0,
                                           t_stop=1,
                                           full_tchans=4) is mismatched
    assert file_mutation._evaluate_t_profile_values(frame, 3) == 3
    assert_allclose(
        file_mutation._evaluate_t_profile_values(frame,
                                                 lambda ts: 2,
                                                 integrate_t_profile=True,
                                                 t_subsamples=2),
        np.full(frame.tchans, 2),
    )
    assert_allclose(file_mutation._evaluate_t_profile_values(frame, lambda ts: 4),
                    np.full(frame.tchans, 4))
    assert file_mutation._choose_chunk_tchans(frame,
                                              affected_fchans=8,
                                              max_chunk_bytes=None,
                                              chunk_tchans=2) == 2
    with pytest.raises(ValueError, match="chunk_tchans"):
        file_mutation._choose_chunk_tchans(frame,
                                           affected_fchans=8,
                                           max_chunk_bytes=None,
                                           chunk_tchans=0)
    with pytest.raises(ValueError, match="max_chunk_bytes"):
        file_mutation._choose_chunk_tchans(frame,
                                           affected_fchans=8,
                                           max_chunk_bytes=0,
                                           chunk_tchans=None)

    source_path = tmp_path / "source.h5"
    frame.save_hdf5(source_path)
    with stg.Frame.open_copy(source_path, tmp_path / "out.h5") as backed:
        with pytest.raises(ValueError, match="smearing_subsamples"):
            backed.add_signal(path=stg.constant_path(frame.get_frequency(4), drift_rate=0),
                              t_profile=1,
                              f_profile=stg.box_f_profile(width=frame.df),
                              doppler_smearing=True,
                              smearing_subsamples=0)
        empty = backed.add_signal(path=stg.constant_path(frame.get_frequency(4), drift_rate=0),
                                  t_profile=1,
                                  f_profile=stg.box_f_profile(width=frame.df),
                                  bounding_f_range=(frame.get_frequency(-10),
                                                    frame.get_frequency(-9)))
        assert empty.time_chunks == 0

        result = backed.add_signal(path=lambda ts: frame.get_frequency(4),
                                   t_profile=1,
                                   f_profile=stg.box_f_profile(width=frame.df),
                                   bounding_f_range=(frame.get_frequency(3),
                                                     frame.get_frequency(5)))
        assert result.time_chunks > 0


def test_private_context_and_io_helpers():
    class SourceNoParams:
        metadata = {"science": "kept", "file_backed": True}
        header = None
        shape = (1, 1)

    class Target:
        def __init__(self):
            self.metadata = {}
            self.header = {"source_name": "T",
                           "tsamp": 1,
                           "tstart": 1,
                           "nchans": 1,
                           "nifs": 1,
                           "fch1": 1,
                           "foff": 1}
            self.source_name = "Target"
            self.dt = 2
            self.t_start = 0
            self.fchans = 1
            self.fch1 = 6e9
            self.df = 1
            self.ascending = True

        def add_metadata(self, metadata):
            self.metadata.update(metadata)

    target = Target()
    _finalize_derived_frame(SourceNoParams(), target, operation="unit")
    assert target.metadata["science"] == "kept"
    assert target.metadata["derived"]["operation"] == "unit"

    copied = Target()
    _copy_frame_context(SourceNoParams(), copied)
    assert copied.metadata["science"] == "kept"

    class BadClose:
        def close(self):
            raise RuntimeError("close failed")

    class BadId:
        @property
        def valid(self):
            raise RuntimeError("invalid")

    class Container:
        h5 = BadClose()

    class Waterfall:
        container = Container()

    _close_waterfall_handles(Waterfall())
    Waterfall.container.h5.id = BadId()
    assert _has_live_h5_handle(Waterfall()) is True


def test_spectrogram_backend_edge_contracts(tmp_path):
    with pytest.raises(ValueError, match="Unsupported"):
        open_spectrogram(tmp_path / "bad.txt")

    source = tmp_path / "source.h5"
    stg.Frame(tchans=2, fchans=4).save_hdf5(source)
    copied = tmp_path / "copied.h5"
    copy_spectrogram(source, copied)
    with pytest.raises(FileExistsError):
        copy_spectrogram(source, copied)

    with open_spectrogram(source) as backend:
        assert backend.shape == (2, 4)
    backend = open_spectrogram(source)
    backend.__exit__(None, None, None)

    with open_spectrogram(source) as backend:
        with pytest.raises(IndexError, match="frequency"):
            backend.read_region(0, 1, -1, 1)
        with pytest.raises(IndexError, match="time"):
            backend.read_region(-1, 1, 0, 1)
        with pytest.raises(OSError, match="read-only"):
            backend.write_region(0, 0, np.zeros((1, 1)))

    with open_spectrogram(source, mode="r+") as backend:
        with pytest.raises(ValueError, match="two-dimensional"):
            backend.write_region(0, 0, np.zeros(4))
        with pytest.raises(IndexError, match="time"):
            backend.write_region(-1, 0, np.zeros((1, 1)))
        state = backend.__getstate__()
        assert state["_h5"] is None
        assert state["_dataset"] is None

    fil_source = tmp_path / "source.fil"
    stg.Frame(tchans=2, fchans=4).save_fil(fil_source)
    with open_spectrogram(fil_source) as backend:
        with pytest.raises(IndexError, match="frequency"):
            backend.read_region(0, 1, -1, 1)
        with pytest.raises(IndexError, match="time"):
            backend.read_region(-1, 1, 0, 1)
        with pytest.raises(OSError, match="read-only"):
            backend.write_region(0, 0, np.zeros((1, 1)))

    with open_spectrogram(fil_source, mode="r+") as backend:
        with pytest.raises(ValueError, match="two-dimensional"):
            backend.write_region(0, 0, np.zeros(4))
        with pytest.raises(IndexError, match="time"):
            backend.write_region(-1, 0, np.zeros((1, 1)))
        with pytest.raises(IndexError, match="frequency"):
            backend.write_region(0, 0, np.zeros((1, 5)))
        backend._disk_frequency_slice = lambda f_start, f_stop: (0, 1, False)
        with pytest.raises(ValueError, match="width"):
            backend.write_region(0, 0, np.zeros((1, 2)))

    bad = tmp_path / "bad.fil"
    bad.write_bytes((10).to_bytes(4, "little") + b"HEADER_END")
    with pytest.raises(RuntimeError, match="valid"):
        open_spectrogram(bad)

    with pytest.raises(ValueError, match="Unsupported"):
        _dtype_from_nbits(4)

    import io
    with pytest.raises(RuntimeError, match="Unexpected end"):
        _read_keyword(io.BytesIO(b""))
    old = _HEADER_KEYWORD_TYPES["machine_id"]
    _HEADER_KEYWORD_TYPES["machine_id"] = "bad"
    try:
        keyword = len("machine_id").to_bytes(4, "little") + b"machine_id"
        with pytest.raises(RuntimeError, match="Unsupported"):
            _read_keyword(io.BytesIO(keyword))
    finally:
        _HEADER_KEYWORD_TYPES["machine_id"] = old
