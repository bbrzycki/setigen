import sys
import os
import errno
import numpy as np
from blimpy import read_header, Waterfall


def split_fil_generator(fil_fn, f_window, f_shift=None):
    """
    Creates a generator that returns smaller Waterfall objects by 'splitting'
    an input filterbank file according to the number of frequency samples.

    Parameters
    ----------
    fil_fn : str
        Filterbank filename with .fil extension
    f_window : int
        Number of frequency samples per new filterbank file
    f_shift : int, optional
        Number of samples to shift when splitting filterbank. If
        None, defaults to `f_shift=f_window` so that there is no
        overlap between new filterbank files

    Yields:
    -------
    split : Waterfall
        A blimpy Waterfall object containing a smaller section of the data
    """

    fch1 = read_header(fil_fn)[b'fch1']
    nchans = read_header(fil_fn)[b'nchans']
    df = read_header(fil_fn)[b'foff']

    if f_shift is None:
        f_shift = f_window

    # Note that df is negative!
    f_start = fch1 + f_window * df
    f_stop = fch1

    # Iterates down frequencies, starting from highest
    while f_start >= fch1 + nchans * df:
        split_fil = Waterfall(fil_fn, f_start=f_start, f_stop=f_stop)
        yield split_fil

        f_start += f_shift * df
        f_stop += f_shift * df


def split_fil(fil_fn, output_dir, f_window, f_shift=None):
    """
    Creates a set of new filterbank files by 'splitting' an input filterbank
    file according to the number of frequency samples.

    Parameters
    ----------
    fil_fn : str
        Filterbank filename with .fil extension
    output_dir : str
        Directory for new filterbank files
    f_window : int
        Number of frequency samples per new filterbank file
    f_shift : int, optional
        Number of samples to shift when splitting filterbank. If
        None, defaults to `f_shift=f_window` so that there is no
        overlap between new filterbank files

    Returns
    -------
    split_fns : list of str
        List of new filenames
    """
    if output_dir[-1] != '/':
        output_dir = output_dir + '/'

    try:
        os.makedirs(output_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    split_generator = split_fil_generator(fil_fn,
                                          f_window,
                                          f_shift=f_shift)

    # Iterates down frequencies, starting from highest
    split_fns = []
    for i, split_fil in enumerate(split_generator):
        output_fn = output_dir + '%s_%04d.fil' % (f_window, i)
        split_fil.write_to_fil(output_fn)
        split_fns.append(output_fn)
        print('Saved %s' % output_fn)
    return split_fns


def split_array(data, f_sample_num=None, t_sample_num=None,
                f_shift=None, t_shift=None,
                f_trim=False, t_trim=False):
    """
    Splits NumPy arrays into a list of smaller arrays according to limits in
    frequency and time. This doesn't reduce/combine data, it simply cuts the
    data into smaller chunks.

    Parameters
    ----------
    data : ndarray
        Time-frequency data

    Returns
    -------
    split_data : list of ndarray
        List of new time-frequency data frames
    """

    split_data = []

    if not isinstance(data, np.ndarray):
        sys.exit("Input data must be a NumPy array!")

    height, width = data.shape

    if f_sample_num is None:
        f_sample_num = width
    if t_sample_num is None:
        t_sample_num = height

    if f_shift is None:
        f_shift = f_sample_num
    elif f_shift <= 0:
        sys.exit("Invalid x-direction shift!")

    if t_shift is None:
        t_shift = t_sample_num
    elif t_shift <= 0:
        sys.exit("Invalid y-direction shift!")

    # Save first frame, regardless of overstepping bounds
    y_start = 0
    y_stop = min(t_sample_num, height)
    x_start = 0
    x_stop = min(f_sample_num, width)
    split_data.append(data[y_start:y_stop, x_start:x_stop])
    y_in_bound = (y_stop < height)
    x_in_bound = (x_stop < width)

    # As long as either bound is valid, continue adding frames
    while y_in_bound or x_in_bound:

        # Shift frames in the x direction
        while x_in_bound:
            x_start = x_start + f_shift
            x_stop = min(x_stop + f_shift, width)
            split_data.append(data[y_start:y_stop, x_start:x_stop])
            x_in_bound = (x_stop < width)

        # Break when both y and x are out of bounds
        if not y_in_bound:
            break

        # Shift frames in the y direction and reset x indices
        y_start = y_start + t_shift
        y_stop = min(y_stop + t_shift, height)
        x_start = 0
        x_stop = min(f_sample_num, width)
        split_data.append(data[y_start:y_stop, x_start:x_stop])
        y_in_bound = (y_stop < height)
        x_in_bound = (x_stop < width)

    # Filter out frames that aren't the same specied size
    if t_trim:
        split_data = list(filter(lambda A: A.shape[0] == t_sample_num,
                                 split_data))
    if f_trim:
        split_data = list(filter(lambda A: A.shape[1] == f_sample_num,
                                 split_data))
    return split_data
