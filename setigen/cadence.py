import collections
import numpy as np
import blimpy as bl
import matplotlib.pyplot as plt

from . import frame as _frame
from . import frame_utils


class Cadence(collections.abc.MutableSequence):
    
    def __init__(self,
                 frame_list=None, 
                 t_slew=0,
                 t_overwrite=True):
        self.frames = list()
        # Insert all initialized items, performing type checks
        if not frame_list is None:
            self.extend(frame_list)
        self.t_slew = t_slew
        self.t_overwrite = t_overwrite
        
        
    @property
    def t_start(self):
        if len(self.frames) == 0:
            return None
        return self.frames[0].t_start
        
    @property
    def fch1(self):
        if len(self.frames) == 0:
            return None
        return self.frames[0].fch1
        
    @property
    def ascending(self):
        if len(self.frames) == 0:
            return None
        return self.frames[0].ascending
        
    @property
    def fmin(self):
        if len(self.frames) == 0:
            return None
        return self.frames[0].fmin
        
    @property
    def fmax(self):
        if len(self.frames) == 0:
            return None
        return self.frames[0].fmax
        
    @property
    def df(self):
        if len(self.frames) == 0:
            return None
        return self.frames[0].df
        
    @property
    def dt(self):
        if len(self.frames) == 0:
            return None
        return self.frames[0].dt
        
    @property
    def fchans(self):
        if len(self.frames) == 0:
            return None
        return self.frames[0].fchans
    
    @property
    def tchans(self):
        if len(self.frames) == 0:
            return None
        return sum([frame.tchans 
                    for frame in self.frames])
        

    def check(self, v):
        if not isinstance(v, _frame.Frame):
            raise TypeError(f"{v} is not a Frame object.")
        if len(self.frames) > 0:
            for attr in ['df', 'dt', 'fchans', 'fmin']:
                if getattr(v, attr) != getattr(self.frames[0], attr):
                    raise AttributeError(f"{attr}={getattr(v, attr)} does not match cadence ({getattr(self.frames[0], attr)})")
                

    def __len__(self): 
        return len(self.frames)

    def __getitem__(self, i): 
        if isinstance(i, slice):
            return self.__class__(self.frames[i])
        elif isinstance(i, (list, np.ndarray, tuple)):
            return self.__class__(np.array(self.frames)[i])
        else:
            return self.frames[i]

    def __delitem__(self, i): 
        del self.frames[i]

    def __setitem__(self, i, v):
        self.check(v)
        self.frames[i] = v

    def insert(self, i, v):
        self.check(v)
        self.frames.insert(i, v)

    def __str__(self):
        return str(self.frames)
    
    def overwrite_times(self):
        """
        Overwrite the starting time and time arrays used to compute signals
        in each frame of the cadence, using slew_time (s) to space between 
        frames.
        """
        for i, frame in enumerate(self.frames[1:]):
            frame.t_start = self.frames[i].t_stop + self.t_slew
            
    @property
    def slew_times(self):
        return np.array([self.frames[i].t_start - self.frames[i - 1].t_stop 
                         for i in range(1, len(self.frames))])
    
    def add_signal(self,
                   *args,
                   **kwargs):
        for frame in self.frames:
            frame.ts += frame.t_start - self.t_start
            frame.add_signal(*args, **kwargs)
            frame.ts -= frame.t_start - self.t_start
        
    def apply(self, func):
        return [func(frame) for frame in self.frames]
    
    def plot(self):
        plot_cadence(self)
        
    def consolidate(self):
        """
        Convert full cadence into a single Frame.
        """
        if len(self.frames) == 0:
            return None
        c_frame = _frame.Frame(fchans=self.fchans,
                               tchans=self.tchans,
                               df=self.df,
                               dt=self.dt,
                               fch1=self.fch1,
                               ascending=self.ascending,
                               t_start=self.t_start)
        
        c_frame.data = np.concatenate([frame.data 
                                       for frame in self.frames],
                                      axis=0)
        c_frame.ts = np.concatenate([frame.ts + frame.t_start 
                                      for frame in self.frames],
                                     axis=0)
        return c_frame
        
        
        
def plot_waterfall(frame, f_start=None, f_stop=None, **kwargs):
    """
    Version of blimpy.stax plot_waterfall method without normalization.
    """
    MAX_IMSHOW_POINTS = (4096, 1268)
    from blimpy.utils import rebin
    
    # Load in the data from fil
    plot_f, plot_data = frame.get_waterfall().grab_data(f_start=f_start, f_stop=f_stop)
    if not frame.ascending:
        plot_f = plot_f[::-1]
        plot_data = plot_data[:, ::-1]

    # Make sure waterfall plot is under 4k*4k
    dec_fac_x, dec_fac_y = 1, 1

    # rebinning data to plot correctly with fewer points
    try:
        if plot_data.shape[0] > MAX_IMSHOW_POINTS[0]:
            dec_fac_x = plot_data.shape[0] / MAX_IMSHOW_POINTS[0]
        if plot_data.shape[1] > MAX_IMSHOW_POINTS[1]:
            dec_fac_y =  int(np.ceil(plot_data.shape[1] /  MAX_IMSHOW_POINTS[1]))
        plot_data = rebin(plot_data, dec_fac_x, dec_fac_y)
    except Exception as ex:
        print("\n*** Oops, grab_data returned plot_data.shape={}, plot_f.shape={}"
              .format(plot_data.shape, plot_f.shape))
        print("Waterfall info for {}:".format(wf.filename))
        wf.info()
        raise ValueError("*** Something is wrong with the grab_data output!") from ex

    # determine extent of the plotting panel for imshow
    nints = plot_data.shape[0]
    bottom = (nints - 1) * frame.dt # in seconds
    extent=(plot_f[0], # left
            plot_f[-1], # right
            bottom, # bottom
            0.0) # top

    # plot and scale intensity (log vs. linear)
    kwargs["cmap"] = kwargs.get("cmap", "viridis")
    plot_data = frame_utils.db(plot_data)

    # display the waterfall plot
    this_plot = plt.imshow(plot_data,
        aspect="auto",
        rasterized=True,
        interpolation="nearest",
        extent=extent,
        **kwargs
    )

    # add source name
    ax = plt.gca()
    plt.text(0.03, 0.8, frame.source_name, transform=ax.transAxes, bbox=dict(facecolor="white"))

    return this_plot



def plot_cadence(cadence):
    """
    Collect real observations of blc1 candidate, and produce synthetic copies using extracted
    properties. Plot real and synthetic cadences side-by-side for easy comparison.
    """
    height_ratios = []
    for i, frame in enumerate(cadence):
        height_ratios.append(frame.tchans * frame.dt)
        
    F_min = frame.fmin / 1e6
    F_max = frame.fmax / 1e6

    # Compute the midpoint frequency for the x-axis.
    F_mid = np.abs(F_min + F_max) / 2
    
    # Create plot grid
    n_plots = len(cadence)
    fig_array, axs = plt.subplots(nrows=n_plots,
                                  ncols=1,
                             sharex=True,
                             sharey=False, 
                             dpi=200,
                             figsize=(12, 2*n_plots),
                             gridspec_kw={"height_ratios" : height_ratios})
    
    
    # Iterate over data for min/max values, real
    for i, frame in enumerate(cadence):
        data = frame.data
        if i == 0:
            px_min = np.min(data)
            px_max = np.max(data)
        else:
            if px_min > np.min(data):
                px_min = np.min(data)
            if px_max < np.max(data):
                px_max = np.max(data)
    
    # Plot real observations
    for i, frame in enumerate(cadence):
        plt.sca(axs[i])
        if i == 0:
            plt.title(f"Source: {frame.source_name}")
            
        last_plot = plot_waterfall(frame, vmin=frame_utils.db(px_min), vmax=frame_utils.db(px_max))
        
    factor = 1e6
    units = "Hz"
    xloc = np.linspace(F_min, F_max, 5)
    xticks = [round(loc_freq) for loc_freq in (xloc - F_mid) * factor]
    if np.max(xticks) > 1000:
        xticks = [xt / 1000 for xt in xticks]
        units = "kHz"
    plt.xticks(xloc, xticks)
    plt.xlabel("Relative Frequency [%s] from %f MHz" % (units, F_mid))
    plt.ylabel("Time [s]")
    
    # Adjust plots
    plt.subplots_adjust(hspace=0.02, wspace=0.1)    
        
    # Add colorbar.
    cax = fig_array.add_axes([0.94, 0.11, 0.03, 0.77])
    fig_array.colorbar(last_plot, cax=cax, label="Power (dB)")