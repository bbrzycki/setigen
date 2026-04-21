import pytest
import copy
import matplotlib.pyplot as plt

from astropy import units as u
import setigen as stg
from astropy.time import Time
from setigen._plot.axes import _get_extent_units


@pytest.fixture()
def t_start_setup():
    mjd_start = 60000
    obs_length = 300
    slew_time = 15

    t_start_arr = [Time(mjd_start, format='mjd').unix]
    for i in range(1, 6):
        t_start_arr.append(t_start_arr[i - 1] + obs_length + slew_time)
    return t_start_arr


@pytest.fixture()
def cadence_setup(t_start_setup):
    frame_list = []
    for i in range(6):
        frame_list.append(stg.Frame(fchans=256, 
                                    tchans=16, 
                                    t_start=t_start_setup[i], 
                                    source_name=f"Obs{i}",
                                    seed=i))
        
    cad = stg.Cadence(frame_list=frame_list)
    cad.apply(lambda f: f.add_noise(1))
    return cad


@pytest.fixture()
def frame_setup():
    frame = stg.Frame(fchans=1024,
                      tchans=32,
                      df=2.7939677238464355,
                      dt=18.253611008,
                      fch1=6095.214842353016*u.MHz,
                      seed=0)
    frame.add_noise(1)
    return frame


def test_plot_extents():
    frame = stg.Frame(fchans=3e3,
                      tchans=16,
                      df=1e6,
                      dt=18.253611008)
    assert _get_extent_units(frame) == (1e9, "GHz")
    frame = stg.Frame(fchans=3e3,
                      tchans=16,
                      df=1e3,
                      dt=18.253611008)
    assert _get_extent_units(frame) == (1e6, "MHz")
    frame = stg.Frame(fchans=1024,
                      tchans=16,
                      df=3,
                      dt=18.253611008)
    assert _get_extent_units(frame) == (1e3, "kHz")


def test_plot_frame_options(frame_setup, recwarn):
    frame = copy.deepcopy(frame_setup)
    assert frame.plot(ftype="fmid") is not None
    assert frame.plot(ftype="fmin", colorbar=False) is not None
    assert frame.plot(ftype="f", minor_ticks=True, label=True) is not None
    assert frame.plot(ftype="px", db=False, grid=True) is not None
    assert frame.plot(ftype="fmid", ttype="px", swap_axes=True) is not None
    assert frame.plot(ftype="px", ttype="trel", swap_axes=True) is not None
    assert frame.plot(ftype="bins", ttype="bins") is not None
    assert plt.gca().get_xlabel() == "Frequency (bins)"
    assert plt.gca().get_ylabel() == "Time (bins)"
    assert not any("FigureCanvasAgg is non-interactive" in str(w.message)
                   for w in recwarn.list)


def test_plot_cadence_options(cadence_setup, recwarn):
    cad = copy.deepcopy(cadence_setup)
    axs, cax = cad.plot(ftype="fmid", slew_times=True)
    assert axs is not None
    assert cax is not None
    axs = cad.plot(ftype="fmin", colorbar=False, title=True)
    assert axs is not None
    axs, cax = cad.plot(ftype="f", minor_ticks=True, labels=False)
    assert axs is not None
    assert cax is not None
    axs, cax = cad.plot(ftype="px", db=False, grid=True)
    assert axs is not None
    assert cax is not None
    axs, cax = cad.plot(ftype="fmid", ttype="px")
    assert axs is not None
    assert cax is not None
    axs, cax = cad.plot(ftype="px", ttype="trel")
    assert axs is not None
    assert cax is not None
    axs, cax = cad.plot(ftype="bins", ttype="bins")
    assert axs.shape[0] == 2 * len(cad) - 1
    assert cax is not None
    assert not any("FigureCanvasAgg is non-interactive" in str(w.message)
                   for w in recwarn.list)


def test_spectrum_plot_labels():
    spectrum = stg.Spectrum(fchans=16,
                            df=1e3,
                            dt=1,
                            fch1=6 * u.GHz,
                            seed=0)
    spectrum.add_noise(1)

    spectrum.plot(ftype="bins")
    assert plt.gca().get_xlabel() == "Frequency (bins)"
    assert plt.gca().get_ylabel() == "Integrated Power (Arbitrary Units)"
    plt.close("all")

    spectrum.plot(ftype="fmid", db=True)
    assert "Relative Frequency" in plt.gca().get_xlabel()
    assert plt.gca().get_ylabel() == "Integrated Power (dB)"
    plt.close("all")


def test_timeseries_plot_labels():
    timeseries = stg.TimeSeries(tchans=16,
                                df=1,
                                dt=2,
                                fch1=6 * u.GHz,
                                seed=0)
    timeseries.add_noise(1)

    timeseries.plot(ttype="bins")
    assert plt.gca().get_xlabel() == "Time (bins)"
    assert plt.gca().get_ylabel() == "Integrated Power (Arbitrary Units)"
    plt.close("all")

    timeseries.plot(ttype="trel", db=True)
    assert plt.gca().get_xlabel() == "Time (s)"
    assert plt.gca().get_ylabel() == "Integrated Power (dB)"
    plt.close("all")
