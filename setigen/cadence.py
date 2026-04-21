from __future__ import annotations

import collections
import numpy as np
import pickle
from typing import Any, Callable, Iterable

from ._constants import ORDER_LABEL_METADATA_KEY
from . import frame as _frame
from . import plots
from . import utils
from ._typing import PathLike

_CADENCE_COMPATIBILITY_ATTRS = ("df", "dt", "fchans", "fmin")


class Cadence(collections.abc.MutableSequence):
    """Organize a sequence of compatible `Frame` objects into a cadence."""
    def __init__(self,
                 frame_list: Iterable[_frame.Frame] | None = None,
                 t_slew: float = 0,
                 t_overwrite: bool = False) -> None:
        """Initialize a cadence.

        Args:
            frame_list: Optional iterable of frames to include.
            t_slew: Slew time between frames in seconds.
            t_overwrite: Whether to overwrite frame start times to enforce slew
                spacing.
        """
        self.frames = list()
        # Insert all initialized items, performing type _checks
        if not frame_list is None:
            self.extend(frame_list)
        
        self.t_slew = t_slew
        self.t_overwrite = t_overwrite
        if t_overwrite:
            self.overwrite_times()
        
    @property
    def t_start(self) -> float | None:
        """Return the cadence start time in Unix seconds."""
        if len(self.frames) == 0:
            return None
        return self.frames[0].t_start
        
    @property
    def fch1(self) -> float | None:
        """Return the first-channel frequency shared by the cadence."""
        if len(self.frames) == 0:
            return None
        return self.frames[0].fch1
        
    @property
    def ascending(self) -> bool | None:
        """Return whether cadence frequencies are ascending."""
        if len(self.frames) == 0:
            return None
        return self.frames[0].ascending
        
    @property
    def fmin(self) -> float | None:
        """Return the minimum cadence frequency."""
        if len(self.frames) == 0:
            return None
        return self.frames[0].fmin
        
    @property
    def fmax(self) -> float | None:
        """Return the maximum cadence frequency."""
        if len(self.frames) == 0:
            return None
        return self.frames[0].fmax

    @property 
    def fmid(self) -> float | None:
        """Return the midpoint cadence frequency."""
        if len(self.frames) == 0:
            return None 
        return self.frames[0].fmid
        
    @property
    def df(self) -> float | None:
        """Return the shared cadence frequency resolution."""
        if len(self.frames) == 0:
            return None
        return self.frames[0].df
        
    @property
    def dt(self) -> float | None:
        """Return the shared cadence time resolution."""
        if len(self.frames) == 0:
            return None
        return self.frames[0].dt
        
    @property
    def fchans(self) -> int | None:
        """Return the number of frequency channels in each frame."""
        if len(self.frames) == 0:
            return None
        return self.frames[0].fchans
    
    @property
    def tchans(self) -> int | None:
        """Return the total number of time channels across the cadence."""
        if len(self.frames) == 0:
            return None
        return sum([frame.tchans 
                    for frame in self.frames])
    @property
    def obs_range(self) -> float | None:
        """Return the total observed time span covered by the cadence."""
        if len(self.frames) == 0:
            return None
        return self.frames[-1].t_stop - self.frames[0].t_start
        
    def _check(self, v: _frame.Frame) -> None:
        """Validate that a frame is cadence-compatible.

        Args:
            v: Frame to validate.

        Raises:
            TypeError: If the object is not a frame.
            AttributeError: If the frame is incompatible with existing cadence
                members.
        """
        if not isinstance(v, _frame.Frame):
            raise TypeError(f"{v} is not a Frame object.")
        if len(self.frames) > 0:
            for attr in _CADENCE_COMPATIBILITY_ATTRS:
                if getattr(v, attr) != getattr(self.frames[0], attr):
                    raise AttributeError(f"{attr}={getattr(v, attr)} does not match cadence ({getattr(self.frames[0], attr)})")
                
    def __len__(self) -> int: 
        return len(self.frames)

    def __getitem__(self, i: Any) -> _frame.Frame | "Cadence": 
        if isinstance(i, slice):
            return self.__class__(self.frames[i])
        elif isinstance(i, (list, np.ndarray, tuple)):
            return self.__class__(np.array(self.frames)[i])
        else:
            return self.frames[i]

    def __delitem__(self, i: int) -> None: 
        del self.frames[i]

    def __setitem__(self, i: int, v: _frame.Frame) -> None:
        self._check(v)
        self.frames[i] = v

    def insert(self, i: int, v: _frame.Frame) -> None:
        """Insert a compatible frame into the cadence.

        Args:
            i: Insertion index.
            v: Frame to insert.
        """
        self._check(v)
        self.frames.insert(i, v)

    def __str__(self) -> str:
        return str(self.frames)
    
    def overwrite_times(self) -> None:
        """Overwrite frame start times using the configured slew spacing."""
        for i, frame in enumerate(self.frames[1:]):
            frame.t_start = self.frames[i].t_stop + self.t_slew
            
    @property
    def slew_times(self) -> np.ndarray:
        """Return slew times between consecutive frames."""
        return np.array([self.frames[i].t_start - self.frames[i - 1].t_stop 
                         for i in range(1, len(self.frames))])
    
    def add_signal(self, *args: Any, **kwargs: Any) -> None:
        """Add the same signal specification to every frame in the cadence.

        Args:
            *args: Positional arguments forwarded to `Frame.add_signal()`.
            **kwargs: Keyword arguments forwarded to `Frame.add_signal()`.
        """
        for frame in self.frames:
            frame.ts += frame.t_start - self.t_start
            frame.add_signal(*args, **kwargs)
            frame.ts -= frame.t_start - self.t_start
        
    def apply(self, func: Callable[[_frame.Frame], Any]) -> list[Any]:
        """Apply a function to each frame in the cadence.

        Args:
            func: Callable that accepts a frame.

        Returns:
            Results returned from each frame application.
        """
        return [func(frame) for frame in self.frames]
    
    @utils._copy_docstring(plots.plot_cadence)
    def plot(self, *args: Any, **kwargs: Any) -> Any:
        return plots.plot_cadence(self, *args, **kwargs)
        
    def consolidate(self) -> _frame.Frame | None:
        """Concatenate the cadence into a single frame.

        Returns:
            Consolidated frame, or `None` when the cadence is empty.
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

    def save_pickle(self, filename: PathLike) -> None:
        """Serialize the full cadence with pickle.

        Args:
            filename: Output pickle path.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load_pickle(cls, filename: PathLike) -> "Cadence":
        """Load a cadence from a pickled file.

        Args:
            filename: Input pickle path created by `save_pickle()`.

        Returns:
            Deserialized cadence object.
        """
        with open(filename, 'rb') as f:
            cad = pickle.load(f)
        return cad
        
        
class OrderedCadence(Cadence):
    """Cadence variant that tracks per-frame order labels."""
    def __init__(self,
                 frame_list: Iterable[_frame.Frame] | None = None,
                 order: str = "ABACAD",
                 t_slew: float = 0,
                 t_overwrite: bool = False) -> None:
        """Initialize an ordered cadence.

        Args:
            frame_list: Optional iterable of frames to include.
            order: Cadence order string such as `"ABACAD"`.
            t_slew: Slew time between frames in seconds.
            t_overwrite: Whether to overwrite frame start times to enforce slew
                spacing.
        """
        self.order = order
        Cadence.__init__(self, 
                         frame_list=frame_list, 
                         t_slew=t_slew, 
                         t_overwrite=t_overwrite)

    def __setitem__(self, i: int, v: _frame.Frame) -> None:
        self._check(v)
        if i < 0:
            i = len(self) + i
        if ORDER_LABEL_METADATA_KEY not in v.metadata:
            v.add_metadata({ORDER_LABEL_METADATA_KEY: self.order[i]})
        self.frames[i] = v

    def insert(self, i: int, v: _frame.Frame) -> None:
        """Insert a frame and assign its order label when needed.

        Args:
            i: Insertion index.
            v: Frame to insert.
        """
        self._check(v)
        if i < 0:
            i = len(self) + i
        if ORDER_LABEL_METADATA_KEY not in v.metadata:
            v.add_metadata({ORDER_LABEL_METADATA_KEY: self.order[i]})
        self.frames.insert(i, v)

    def set_order(self, order: str) -> None:
        """Reassign cadence order labels.

        Args:
            order: New cadence order string.
        """
        self.order = order 
        for i, fr in enumerate(self.frames):
            fr.add_metadata({ORDER_LABEL_METADATA_KEY: self.order[i]})

    def by_label(self, order_label: str = "A") -> Cadence:
        """Filter frames by cadence order label.

        Args:
            order_label: Cadence label to match.

        Returns:
            New cadence containing only matching frames.
        """
        return Cadence(frame_list=[frame for frame in self 
                                   if frame.metadata[ORDER_LABEL_METADATA_KEY] == order_label])
