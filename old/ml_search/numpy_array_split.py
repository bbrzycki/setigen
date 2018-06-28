import sys
import numpy as np
import pandas as pd
from blimpy import Waterfall

# fil = Waterfall(input_fn)
# data = np.squeeze(fil.data)
# print(split_data(data, x_sample_num = 1024, y_sample_num = None))

def split_array(data, x_sample_num=None, y_sample_num=None,
                     x_shift=None, y_shift=None,
                     x_trim=False, y_trim=False):
    """
    Splits NumPy arrays into a list of smaller arrays according to limits in frequency and time.
    This doesn't reduce/combine data, it simply cuts the data into smaller chunks.

    Args:
        data, NumPy array (ndarray)

    Return:
        split_data, list of NumPy arrays
    """

    split_data = []

    if not isinstance(data, np.ndarray):
        sys.exit("Input data must be a NumPy array!")

    height, width = data.shape

    if x_sample_num is None:
        x_sample_num = width
    if y_sample_num is None:
        y_sample_num = height

    if x_shift is None:
        x_shift = x_sample_num
    elif x_shift <= 0:
        sys.exit("Invalid x-direction shift!")

    if y_shift is None:
        y_shift = y_sample_num
    elif y_shift <= 0:
        sys.exit("Invalid y-direction shift!")

    # Save first frame, regardless of overstepping bounds
    y_start = 0
    y_stop = min(y_sample_num, height)
    x_start = 0
    x_stop = min(x_sample_num, width)
    split_data.append(data[y_start:y_stop,x_start:x_stop])
    y_in_bound = (y_stop < height)
    x_in_bound = (x_stop < width)

    # As long as either bound is valid, continue adding frames
    while y_in_bound or x_in_bound:

        # Shift frames in the x direction
        while x_in_bound:
            x_start = x_start + x_shift
            x_stop = min(x_stop + x_shift, width)
            split_data.append(data[y_start:y_stop,x_start:x_stop])
            x_in_bound = (x_stop < width)

        # Break when both y and x are out of bounds
        if not y_in_bound:
            break

        # Shift frames in the y direction and reset x indices
        y_start = y_start + y_shift
        y_stop = min(y_stop + y_shift, height)
        x_start = 0
        x_stop = min(x_sample_num, width)
        split_data.append(data[y_start:y_stop,x_start:x_stop])
        y_in_bound = (y_stop < height)
        x_in_bound = (x_stop < width)

    # Filter out frames that aren't the same specied size
    if y_trim:
        split_data = list(filter(lambda A: A.shape[0] == y_sample_num, split_data))
    if x_trim:
        split_data = list(filter(lambda A: A.shape[1] == x_sample_num, split_data))
    return split_data
