import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.offsetbox import AnchoredText

from . import frame_utils


def _get_extent_units(frame):
    """
    Simple function to get best frequency units for plots.
    """
    f_range = np.abs(frame.fmax - frame.fmin)
    if f_range > 2e9:
        return 1e9, "GHz"
    elif f_range > 2e6:
        return 1e6, "MHz" 
    elif f_range > 2e3:
        return 1e3, "kHz" 
    else:
        return 1, "Hz"
    

def _frequency_formatter(frame, ftype):
    if ftype == "fmid":
        def formatter(x, pos):
            return x / _get_extent_units(frame)[0]
            x = x / _get_extent_units(frame)[0]
            return f"{int(x):d}"
    elif ftype == "fmin": 
        def formatter(x, pos):
            return x / _get_extent_units(frame)[0]
            x = x / _get_extent_units(frame)[0]
            return f"{int(x):d}"
    else:
        def formatter(x, pos):
            return x / 1e6
            x = x / 1e6
            return f"{int(x):d}"
    return formatter


def plot_frame(frame, 
               ftype="fmid", 
               db=True, 
               colorbar=True, 
               label=False,
               minor_ticks=False,
               grid=False,
               swap_axes=False,
               **kwargs):
    """
    Plot frame spectrogram data.
    
    Parameters
    ----------
    frame : Frame
        Frame to plot
    ftype : {"fmid", "fmin", "f", "px"}, default: "fmid"
        Types of axis labels, particularly the x-axis. "px" puts axes in units 
        of pixels. The others are all in frequency: "fmid" shows frequencies 
        relative to the central frequency, "fmin" is relative to the minimum 
        frequency, and "f" is absolute frequency.
    db : bool, default: True
        Option to convert intensities to dB
    colorbar : bool, default: True
        Whether to display colorbar
    label : bool, default: False
        Option to place target name as a label in plot
    minor_ticks : bool, default: False
        Option to include minor ticks on both axes
    grid : bool, default: False
        Option to overplot grid from major ticks
    invert_axes : bool, default: False
        Option to swap frequency and time axes

    Return 
    ------
    p : matplotlib.image.AxesImage
        Spectrogram axes object
    """
    # Scale intensity if necessary (log vs. linear)
    data = frame.data
    if db:
        data = frame_utils.db(data)

    # matplotlib extend order is (left, right, bottom, top)
    if ftype == "fmid":
        extent = (
            frame.fmin - frame.fmid - frame.df / 2,
            frame.fmax - frame.fmid + frame.df / 2,
            frame.tchans * frame.dt, 
            0
        )
    elif ftype == "fmin":
        extent = (
            -frame.df / 2,
            frame.fmax - frame.fmin + frame.df / 2,
            frame.tchans * frame.dt, 
            0
        ) 
    elif ftype == "f":
        extent = (
            frame.fmin - frame.df / 2,
            frame.fmax + frame.df / 2,
            frame.tchans * frame.dt, 
            0
        )
    else: 
        # ftype == "px"
        extent = (
            -1 / 2,
            frame.fchans - 1 / 2,
            frame.tchans - 1 / 2, 
            -1 / 2
        ) 

    if swap_axes:
        data = data.T[::-1, :]
        extent = np.array(extent)[[3, 2, 0, 1]]

    # display the waterfall plot
    p = plt.imshow(data,
                   aspect="auto",
                   rasterized=True,
                   interpolation="none",
                   extent=extent,
                   **kwargs)
    if colorbar:
        if db:
            cbar_label = "Power (dB)"
        else:
            cbar_label = "Power (Arbitrary Units)"
        cbar = plt.colorbar()
        cbar.ax.set_ylabel(cbar_label)
            
    # Format axes
    ax = plt.gca()
    if swap_axes:
        faxis = ax.yaxis
        taxis = ax.xaxis 
    else:
        faxis = ax.xaxis 
        taxis = ax.yaxis
        
    faxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    if minor_ticks:
        faxis.set_minor_locator(ticker.AutoMinorLocator(n=5))
        taxis.set_minor_locator(ticker.AutoMinorLocator())

    if ftype in ["fmid", "fmin", "f"]:
        tlabel = "Time (s)"
        faxis.set_major_formatter(plt.FuncFormatter(_frequency_formatter(frame, ftype)))
        units = _get_extent_units(frame)[1]
        if ftype == "fmid":
            flabel = f"Relative Frequency ({units}) from {frame.fmid * 1e-6:.6f} MHz"
        elif ftype == "fmin":
            flabel = f"Relative Frequency ({units}) from {frame.fmin * 1e-6:.6f} MHz"
        else:
            # ftype == "f"
            flabel = f"Frequency (MHz)"
    else:
        # ftype == "px"
        flabel = "Frequency (px)"
        tlabel = "Time (px)"

    faxis.set_label_text(flabel)
    taxis.set_label_text(tlabel)
    
    if grid:
        plt.grid(True)
        
    if label:
        if "order_label" in frame.metadata:
            source_label = f'{frame.metadata["order_label"]}: {frame.source_name}'
        else:
            source_label = frame.source_name
                 
        at = AnchoredText(
            source_label,
            loc="upper left",
            frameon=True,
        )
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax.add_artist(at)
    
    return p

                 
def plot_cadence(cadence, 
                 ftype="fmid", 
                 db=True, 
                 slew_times=False,
                 colorbar=True, 
                 labels=True,
                 title=False,
                 minor_ticks=False,
                 grid=False,
                 **kwargs):
    """
    Plot cadence as a multi-panel figure.

    Parameters
    ----------
    cadence : Cadence
        Cadence to plot
    ftype : {"fmid", "fmin", "f", "px"}, default: "fmid"
        Types of axis labels, particularly the x-axis. "px" puts axes in units 
        of pixels. The others are all in frequency: "fmid" shows frequencies 
        relative to the central frequency, "fmin" is relative to the minimum 
        frequency, and "f" is absolute frequency.
    db : bool, default: True
        Option to convert intensities to dB
    slew_times : bool, default: False
        Option to space subplots vertically proportional to slew times
    colorbar : bool, default: True
        Whether to display colorbar
    labels : bool, default: True
        Option to place target name as a label in each subplot
    title : bool, default: False
        Option to place first source name as the figure title
    minor_ticks : bool, default: False
        Option to include minor ticks on both axes
    grid : bool, default: False
        Option to overplot grid from major ticks

    Return 
    ------
    axs : matplotlib.axes.Axes
        Axes subplots
    cax : matplotlib.axes.Axes
        Colorbar axes, if created
    """
    height_ratios = np.zeros(2 * len(cadence) - 1)
    for i, frame in enumerate(cadence):
        height_ratios[2*i] = frame.tchans * frame.dt
        if i != len(cadence) - 1:
            if slew_times:
                if cadence.slew_times[i] < 0:
                    raise ValueError(f"Frame {i + 1} starts after the end of "
                                     f"frame {i}, so we cannot space by slew "
                                     f"times (delta t = {cadence.slew_times[i]:.1f} s)")
                height_ratios[2*i+1] = cadence.slew_times[i]

    # Create plot grid
    fig = plt.gcf()
    axs = fig.subplots(nrows=len(height_ratios),
                       ncols=1,
                       sharex=True,
                       sharey=False, 
                       height_ratios=height_ratios,
                       **kwargs)
    
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
    if db:
        px_min = frame_utils.db(px_min)
        px_max = frame_utils.db(px_max)
    
    # Plot real observations
    for i, frame in enumerate(cadence):
        plt.sca(axs[2*i])
        if title and i == 0:
            plt.title(f"Source: {frame.source_name}")
            
        last_plot = plot_frame(frame, 
                                   ftype=ftype,
                                   db=db,
                                   colorbar=False,
                                   label=labels,
                                   minor_ticks=minor_ticks,
                                   grid=grid,
                                   vmin=px_min, 
                                   vmax=px_max)
        
        if i != len(cadence) - 1:
            plt.xlabel(None)

            plt.sca(axs[2*i+1])
            ax = plt.gca()
            ax.yaxis.set_major_locator(ticker.NullLocator())
        
    plt.subplots_adjust(hspace=0., wspace=0.)    
        
    # Add colorbar
    if colorbar:
        if db:
            cbar_label = "Power (dB)"
        else:
            cbar_label = "Power (Arbitrary Units)"
        cax = fig.add_axes([0.94, 0.11, 0.03, 0.77])
        fig.colorbar(last_plot, cax=cax, label=cbar_label)
        return axs, cax
    else:
        return axs