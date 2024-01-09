import numpy as np
from astropy import units as u
import matplotlib.pyplot as plt

from . import frame 


class TimeSeries(frame.Frame):
    """
    A class to store an intensity time series as Frame object.
    """
    def __init__(self,
                 tchans=None,
                 df=2.7939677238464355*u.Hz,
                 dt=18.253611008*u.s,
                 fch1=6*u.GHz,
                 ascending=False,
                 data=None,
                 seed=None,
                 **kwargs):
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

    def array(self, db=False):
        return self.get_data(db=db)[:, 0]

    def plot(self, 
             ttype="trel",
             norm=False,
             db=False,
             minor_ticks=False,
             **kwargs):
        """
        Plot spectrum.

        Parameters
        ----------
        ttype : {"trel", "px", "bins"}, default: "trel"
            Type of time axis labels. "px" and "bins" put the axis in units of 
            pixels (bins), and "trel" sets the axis in time units relative to 
            the start.
        norm : bool, default: False 
            Option to plot time series normalized to a mean of 1
        db : bool, default: False
            Option to convert intensities to dB
        minor_ticks : bool, default: False
            Option to include minor ticks on both axes
        """
        if norm:
            new_ts = self.copy()
            new_ts.normalize() 
            ip = new_ts.array(db=db)
        else:
            ip = self.array(db=db)

        if ttype == "trel":
            ts = self.ts
        else:
            # ttype == "px" or "bins"
            ts = (self.ts - self.ts[0]) / self.dt

        plt.plot(ts, ip, **kwargs)

        ax = plt.gca()
        taxis = ax.xaxis 
        if minor_ticks:
            taxis.set_minor_locator(ticker.AutoMinorLocator())

        if ttype == "same":
            if ftype in ["fmid", "fmin", "f"]:
                tlabel = "Time (s)"
            else:
                tlabel = f"Time ({ftype})"
        elif ttype == "trel":
            tlabel = "Time (s)"
        else:
            # ttype == "px" or "bins"
            tlabel = f"Time ({ttype})"

        if db:
            ylabel = "Integrated Power (dB)"
        else:
            ylabel = "Integrated Power (Arbitrary Units)"

        taxis.set_label_text(tlabel)
        ax.yaxis.set_label_text(ylabel)

    def normalize(self):
        """
        Normalize time series to mean 1.
        """
        self.data = self.data / np.mean(self.data)

    def autocorr(self, remove_spike=False):
        """
        Calculate full autocorrelation, normalizing time series to zero mean and unit variance.
        """
        ts = self.array()
        ts = ts - np.mean(ts)
        acf = np.correlate(ts, ts, 'full')[-len(ts):]
        if remove_spike:
            acf[0] = acf[1]
        acf /= acf[0] # This is essentially the variance (scaled by len(ts))
        return acf


    def acf(self, remove_spike=False):
        return self.autocorr(remove_spike=remove_spike)