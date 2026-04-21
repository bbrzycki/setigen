from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.offsetbox import AnchoredText
from typing import Any

from ._constants import ORDER_LABEL_METADATA_KEY
from . import utils
from ._plot.axes import (
    _ResolvedAxisSpec,
    _frequency_formatter,
    _get_frame_frequency_edges,
    _get_frame_time_edges,
    _get_frequency_axis_label,
    _get_time_axis_label,
)


def plot_frame(frame: Any, 
               ftype: str="fmid", 
               ttype: str="same",
               db: bool=True, 
               colorbar: bool=True, 
               label: bool=False,
               minor_ticks: bool=False,
               grid: bool=False,
               swap_axes: bool=False,
               **kwargs: Any) -> Any:
    """Plot frame spectrogram data.

    Args:
        frame: Frame to plot.
        ftype: Frequency-axis display mode.
        ttype: Time-axis display mode.
        db: Whether to convert intensities to dB.
        colorbar: Whether to display the colorbar.
        label: Whether to place the source name as an anchored label.
        minor_ticks: Whether to enable minor ticks.
        grid: Whether to draw the major-tick grid.
        swap_axes: Whether to swap frequency and time axes.
        **kwargs: Additional `matplotlib.pyplot.imshow()` keyword arguments.

    Returns:
        Spectrogram image artist.
    """
    # Scale intensity if necessary (log vs. linear)
    data = frame.data
    if db:
        data = utils.db(data)

    axis_spec = _ResolvedAxisSpec.from_values(ftype=ftype, ttype=ttype)

    f_edge_min, f_edge_max = _get_frame_frequency_edges(frame, axis_spec)
    t_edge_min, t_edge_max = _get_frame_time_edges(frame, axis_spec)

    # Arrange spectrogram plot and data as necessary
    if not swap_axes:
        extent = (f_edge_min, f_edge_max, t_edge_max, t_edge_min)
    else:
        data = data.T[::-1, :]
        extent = (t_edge_min, t_edge_max, f_edge_min, f_edge_max)

    # Display the waterfall plot
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
    if not swap_axes:
        faxis = ax.xaxis 
        taxis = ax.yaxis
    else:
        faxis = ax.yaxis
        taxis = ax.xaxis 
        
    faxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    if minor_ticks:
        faxis.set_minor_locator(ticker.AutoMinorLocator(n=5))
        taxis.set_minor_locator(ticker.AutoMinorLocator())

    if axis_spec.uses_frequency_units:
        faxis.set_major_formatter(plt.FuncFormatter(_frequency_formatter(frame, ftype)))
    flabel = _get_frequency_axis_label(frame, axis_spec)
    tlabel = _get_time_axis_label(axis_spec)

    faxis.set_label_text(flabel)
    taxis.set_label_text(tlabel)
    
    if grid:
        plt.grid(True)
        
    if label:
        if ORDER_LABEL_METADATA_KEY in frame.metadata:
            source_label = f'{frame.metadata[ORDER_LABEL_METADATA_KEY]}: {frame.source_name}'
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

                 
def plot_cadence(cadence: Any, 
                 ftype: str="fmid", 
                 ttype: str="same",
                 db: bool=True, 
                 slew_times: bool=False,
                 colorbar: bool=True, 
                 labels: bool=True,
                 title: bool=False,
                 minor_ticks: bool=False,
                 grid: bool=False,
                 **kwargs: Any) -> tuple[Any, Any | None]:
    """Plot a cadence as a vertically stacked figure.

    Args:
        cadence: Cadence to plot.
        ftype: Frequency-axis display mode.
        ttype: Time-axis display mode.
        db: Whether to convert intensities to dB.
        slew_times: Whether to space panels proportionally to slew time.
        colorbar: Whether to display a shared colorbar.
        labels: Whether to place source labels on each subplot.
        title: Whether to add the first source name as the figure title.
        minor_ticks: Whether to enable minor ticks.
        grid: Whether to draw the major-tick grid.
        **kwargs: Additional `matplotlib.figure.Figure.subplots()` keyword
            arguments.

    Returns:
        Tuple of subplot axes and optional colorbar axis.

    Raises:
        ValueError: If negative slew-time spacing is requested.
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
        px_min = utils.db(px_min)
        px_max = utils.db(px_max)
    
    # Plot real observations
    for i, frame in enumerate(cadence):
        plt.sca(axs[2*i])
        if title and i == 0:
            plt.title(f"Source: {frame.source_name}")
            
        last_plot = plot_frame(frame, 
                                ftype=ftype,
                                ttype=ttype,
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
