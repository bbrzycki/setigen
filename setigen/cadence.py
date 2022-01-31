import collections
from . import frame


class Cadence(collections.MutableSequence):
    
    def __init__(self, iterator_arg=None):
        self.frames = list()
        # Insert all initialized items, performing type checks
        if not iterator_arg is None:
            self.extend(iterator_arg)

    def check(self, v):
        if not isinstance(v, frame.Frame):
            raise TypeError(f"{v} is not a Frame object.")

    def __len__(self): 
        return len(self.frames)

    def __getitem__(self, i): 
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