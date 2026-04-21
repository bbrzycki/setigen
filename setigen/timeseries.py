from __future__ import annotations

import numpy as np
from astropy import units as u
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import Any

from . import frame 
from ._plot.axes import (
    _ResolvedAxisSpec,
    _get_time_axis_label,
    _get_timeseries_x_values,
)


class TimeSeries(frame.Frame):
    """Store a one-dimensional time series as a `Frame` subclass."""
    def __init__(self,
                 tchans: int | None = None,
                 df: Any = 2.7939677238464355*u.Hz,
                 dt: Any = 18.253611008*u.s,
                 fch1: Any = 6*u.GHz,
                 ascending: bool = False,
                 data: np.ndarray | None = None,
                 seed: Any = None,
                 **kwargs: Any) -> None:
        """Initialize a one-channel time-series frame.

        Args:
            tchans: Number of time channels.
            df: Frequency resolution.
            dt: Time resolution.
            fch1: Frequency of the first channel.
            ascending: Whether the frequency axis is ascending.
            data: Optional preloaded time series.
            seed: Random seed or generator.
            **kwargs: Additional frame-construction keyword arguments.
        """
        if "fchans" in kwargs:
            assert kwargs.pop("fchans") == 1
        frame.Frame.__init__(self,
                             fchans=1,
                             tchans=tchans,
                             df=df,
                             dt=dt,
                             fch1=fch1,
                             ascending=ascending,
                             data=data,
                             seed=seed,
                             **kwargs)

    def array(self, db: bool = False) -> np.ndarray:
        """Return the time series as a one-dimensional array.

        Args:
            db: Whether to convert intensities to dB.

        Returns:
            One-dimensional time-series array.
        """
        return self.get_data(db=db)[:, 0]

    def plot(self, 
             ttype: str = "trel",
             norm: bool = False,
             db: bool = False,
             minor_ticks: bool = False,
             **kwargs: Any) -> None:
        """Plot the time series.

        Args:
            ttype: Time-axis display mode.
            norm: Whether to normalize the time series to mean one before
                plotting.
            db: Whether to convert intensities to dB.
            minor_ticks: Whether to enable minor ticks.
            **kwargs: Additional `matplotlib.pyplot.plot()` keyword arguments.
        """
        if norm:
            new_ts = self.copy()
            new_ts.normalize() 
            ip = new_ts.array(db=db)
        else:
            ip = self.array(db=db)

        axis_spec = _ResolvedAxisSpec.from_values(ttype=ttype)
        ts = _get_timeseries_x_values(self, axis_spec)

        plt.plot(ts, ip, **kwargs)

        ax = plt.gca()
        taxis = ax.xaxis 
        if minor_ticks:
            taxis.set_minor_locator(ticker.AutoMinorLocator())

        tlabel = _get_time_axis_label(axis_spec)

        if db:
            ylabel = "Integrated Power (dB)"
        else:
            ylabel = "Integrated Power (Arbitrary Units)"

        taxis.set_label_text(tlabel)
        ax.yaxis.set_label_text(ylabel)

    def normalize(self) -> None:
        """Normalize the time series to unit mean."""
        self.data = self.data / np.mean(self.data)

    def autocorr(self, remove_spike: bool = False) -> np.ndarray:
        """Calculate the normalized autocorrelation of the time series.

        Args:
            remove_spike: Whether to replace the zero-lag spike with the first
                nonzero lag.

        Returns:
            Normalized autocorrelation sequence.
        """
        ts = self.array()
        ts = ts - np.mean(ts)
        acf = np.correlate(ts, ts, 'full')[-len(ts):]
        if remove_spike:
            acf[0] = acf[1]
        acf /= acf[0] # This is essentially the variance (scaled by len(ts))
        return acf

    def acf(self, remove_spike: bool = False) -> np.ndarray:
        """Alias for `autocorr()`.

        Args:
            remove_spike: Whether to replace the zero-lag spike.

        Returns:
            Normalized autocorrelation sequence.
        """
        return self.autocorr(remove_spike=remove_spike)
