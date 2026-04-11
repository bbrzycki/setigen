import numpy as np
from astropy.stats import sigma_clip
from astropy import units as u
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from . import frame 
from ._plot.axes import (
    _ResolvedAxisSpec,
    _frequency_formatter,
    _get_frequency_axis_label,
    _get_spectrum_x_values,
)


class Spectrum(frame.Frame):
    """
    A class to store a frequency spectrum as Frame object.
    """
    def __init__(self,
                 fchans=None,
                 df=2.7939677238464355*u.Hz,
                 dt=18.253611008*u.s,
                 fch1=6*u.GHz,
                 ascending=False,
                 data=None,
                 seed=None,
                 **kwargs):
        if "tchans" in kwargs:
            assert kwargs.pop("tchans") == 1
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

    def array(self, db=False):
        return self.get_data(db=db)[0]

    def plot(self, 
             ftype="fmid",
             snr=False,
             db=False,
             minor_ticks=False,
             **kwargs):
        """
        Plot spectrum.

        Parameters
        ----------
        ftype : {"fmid", "fmin", "f", "px", "bins"}, default: "fmid"
            Type of frequency axis labels. "px" and "bins" put the axis in units of 
            pixels (bins). The others are all in frequency: "fmid" shows frequencies 
            relative to the central frequency, "fmin" is relative to the minimum 
            frequency, and "f" is absolute frequency.
        snr : bool, default: False 
            Option to plot integrated power as signal-to-noise
        db : bool, default: False
            Option to convert intensities to dB
        minor_ticks : bool, default: False
            Option to include minor ticks on both axes
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

    def normalize(self):
        """
        Normalize background to zero mean, unit variance.
        """
        c_data = sigma_clip(self.data)
        self.data = (self.data - np.mean(c_data)) / np.std(c_data)
