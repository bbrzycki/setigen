import collections
import numpy as np

from . import frame as _frame
from . import plots


class Cadence(collections.abc.MutableSequence):
    """
    A class for organizing cadences of Frame objects.

    Parameters
    ----------
    frame_list : list of Frames, optional
        List of frames to be included in the cadence
    t_slew : float, optional
        Slew time between frames
    t_overwrite : bool, optional
        Option to overwrite time data in frames to enforce slew time spacing           
    """
    def __init__(self,
                 frame_list=None, 
                 t_slew=0,
                 t_overwrite=False):
        self.frames = list()
        # Insert all initialized items, performing type _checks
        if not frame_list is None:
            self.extend(frame_list)
        
        self.t_slew = t_slew
        self.t_overwrite = t_overwrite
        if t_overwrite:
            self.overwrite_times()
        
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
    def fmid(self):
        if len(self.frames) == 0:
            return None 
        return self.frames[0].fmid
        
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
    @property
    def obs_range(self):
        return self.frames[-1].t_stop - self.frames[0].t_start
        
    def _check(self, v):
        """
        Ensure that object is a Frame and has consistent parameters with 
        existing frames in the cadence. 
        """
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
        self._check(v)
        self.frames[i] = v

    def insert(self, i, v):
        self._check(v)
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
        """
        Compute slew times in between frames.
        """
        return np.array([self.frames[i].t_start - self.frames[i - 1].t_stop 
                         for i in range(1, len(self.frames))])
    
    def add_signal(self, *args, **kwargs):
        """
        Add signal to each frame in the cadence. Arguments are passed through
        to :code:`Frame.add_signal`.
        """
        for frame in self.frames:
            frame.ts += frame.t_start - self.t_start
            frame.add_signal(*args, **kwargs)
            frame.ts -= frame.t_start - self.t_start
        
    def apply(self, func):
        """
        Apply function to each frame in the cadence.
        """
        return [func(frame) for frame in self.frames]
    
    def plot(self, *args, **kwargs):
        """
        Plot cadence as a multi-panel figure.

        Parameters
        ----------
        cadence : Cadence
            Cadence to plot
        xtype : {"fmid", "fmin", "f", "px"}, default: "fmid"
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
        plots.plot_cadence(self, *args, **kwargs)
        
    def consolidate(self):
        """
        Convert full cadence into a single concatenated Frame.
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
        
        
class OrderedCadence(Cadence):
    """
    A class that extends Cadence for imposing a cadence order, such as 
    "ABACAD" or "ABABAB". Order labels are added to each frame's metadata with
    the :code:`order_label` keyword.

    Parameters
    ----------
    frame_list : list of Frames, optional
        List of frames to be included in the cadence
    order : str, optional
        Cadence order, expressed as a string of letters (e.g. "ABACAD")
    t_slew : float, optional
        Slew time between frames
    t_overwrite : bool, optional
        Option to overwrite time data in frames to enforce slew time spacing           
    """
    def __init__(self,
                 frame_list=None, 
                 order="ABACAD",
                 t_slew=0,
                 t_overwrite=False):
        self.order = order
        Cadence.__init__(self, 
                         frame_list=frame_list, 
                         t_slew=t_slew, 
                         t_overwrite=t_overwrite)

    def __setitem__(self, i, v):
        self._check(v)
        if i < 0:
            i = len(self) + i
        if "order_label" not in v.metadata:
            v.add_metadata({"order_label": self.order[i]})
        self.frames[i] = v

    def insert(self, i, v):
        self._check(v)
        if i < 0:
            i = len(self) + i
        if "order_label" not in v.metadata:
            v.add_metadata({"order_label": self.order[i]})
        self.frames.insert(i, v)

    def set_order(self, order):
        """
        Reassign cadence order.
        """
        self.order = order 
        for i, fr in enumerate(self.frames):
            fr.add_metadata({"order_label": self.order[i]})

    def by_label(self, order_label="A"):
        """
        Filter frames in cadence by their label in the cadence order, specified
        as a letter. Returns matching frames as a new Cadence.
        """
        return Cadence(frame_list=[frame for frame in self 
                                   if frame.metadata["order_label"] == order_label])

