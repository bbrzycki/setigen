from __future__ import annotations

import numpy as np
from astropy.stats import sigma_clip
from astropy import units as u
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import Any

from . import frame 
from ._plot.axes import (
    _ResolvedAxisSpec,
    _frequency_formatter,
    _get_frequency_axis_label,
    _get_spectrum_x_values,
)


class Spectrum(frame.Frame):
    """Store a one-dimensional frequency spectrum as a `Frame` subclass."""
    def __init__(self,
                 fchans: int | None = None,
                 df: Any = 2.7939677238464355*u.Hz,
                 dt: Any = 18.253611008*u.s,
                 fch1: Any = 6*u.GHz,
                 ascending: bool = False,
                 data: np.ndarray | None = None,
                 seed: Any = None,
                 **kwargs: Any) -> None:
        """Initialize a one-row spectral frame.

        Args:
            fchans: Number of frequency channels.
            df: Frequency resolution.
            dt: Time resolution.
            fch1: Frequency of the first channel.
            ascending: Whether the frequency axis is ascending.
            data: Optional preloaded spectrum.
            seed: Random seed or generator.
            **kwargs: Additional frame-construction keyword arguments.
        """
        if "tchans" in kwargs:
            tchans = kwargs.pop("tchans")
            if tchans != 1:
                raise ValueError("Spectrum requires tchans=1")
        if data is not None:
            data = np.asarray(data)
            if data.ndim == 1:
                data = data[np.newaxis, :]
            elif data.ndim != 2 or data.shape[0] != 1:
                raise ValueError("Spectrum data must be one-dimensional or have shape (1, fchans)")
            if fchans is None:
                fchans = data.shape[1]
        frame.Frame.__init__(self,
                             fchans=fchans,
                             tchans=1,
                             df=df,
                             dt=dt,
                             fch1=fch1,
                             ascending=ascending,
                             data=data,
                             seed=seed,
                             **kwargs)

    def array(self, db: bool = False) -> np.ndarray:
        """Return the spectrum as a one-dimensional array.

        Args:
            db: Whether to convert intensities to dB.

        Returns:
            One-dimensional spectral array.
        """
        return self.get_data(db=db)[0]

    def plot(self, 
             ftype: str = "fmid",
             snr: bool = False,
             db: bool = False,
             minor_ticks: bool = False,
             **kwargs: Any) -> None:
        """Plot the spectrum.

        Args:
            ftype: Frequency-axis display mode.
            snr: Whether to plot normalized signal-to-noise instead of raw
                power.
            db: Whether to convert intensities to dB.
            minor_ticks: Whether to enable minor ticks.
            **kwargs: Additional `matplotlib.pyplot.plot()` keyword arguments.
        """
        if snr:
            new_spec = self.copy()
            new_spec.normalize() 
            ip = new_spec.array(db=db)
        else:
            ip = self.array(db=db)

        axis_spec = _ResolvedAxisSpec.from_values(ftype=ftype)
        fs = _get_spectrum_x_values(self, axis_spec)

        plt.plot(fs, ip, **kwargs)

        ax = plt.gca()
        faxis = ax.xaxis 
        # faxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        if minor_ticks:
            faxis.set_minor_locator(ticker.AutoMinorLocator(n=5))

        if axis_spec.uses_frequency_units:
            faxis.set_major_formatter(plt.FuncFormatter(_frequency_formatter(self, ftype)))
        flabel = _get_frequency_axis_label(self, axis_spec)

        if db:
            y_units = "dB"
        else:
            y_units = "Arbitrary Units"
        if snr:
            ylabel = f"S/N ({y_units})"
            if not db:
                ylabel = "S/N"
        else:
            ylabel = f"Integrated Power ({y_units})"

        faxis.set_label_text(flabel)
        ax.yaxis.set_label_text(ylabel)

    def normalize(self) -> None:
        """Normalize the spectrum background to zero mean and unit variance."""
        c_data = sigma_clip(self.data)
        self.data = (self.data - np.mean(c_data)) / np.std(c_data)
