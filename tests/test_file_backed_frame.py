import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_allclose
import pytest

import setigen as stg


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
