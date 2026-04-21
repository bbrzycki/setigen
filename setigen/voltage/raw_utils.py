from __future__ import annotations

import numpy as np
import glob
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any

from .._typing import PathLike


def format_header_line(key: str, value: str | int | float, as_strings: bool = False) -> str:
    """Format a RAW header key-value pair as an 80-character line.

    Args:
        key: Header key.
        value: Header value.
        as_strings: Whether the value is already preformatted as a string.

    Returns:
        Formatted 80-character header line.
    """
    if as_strings:
        if "\'" in value:
            line = f"{key:<8}= {value:<20}"
        else:
            line = f"{key:<8}= {value:>20}"
    else:
        if isinstance(value, str):
            value = f"'{value: <8}'"
            line = f"{key:<8}= {value:<20}"
        else:
            if key == 'TBIN':
                value = f"{value:.14E}"
            line = f"{key:<8}= {value:>20}"
    line = f"{line:<80}"
    return line


def get_header_key_val(header_line: str) -> tuple[str, str]:
    """Split a formatted RAW header line into key and value.

    Args:
        header_line: Formatted header line.

    Returns:
        Header key and value as strings.
    """
    key = header_line[:8].strip()
    value = header_line[9:].strip().strip("''")
    return key, value


def read_header(filename: PathLike) -> dict[str, str]:
    """Read a GUPPI RAW header into a dictionary.

    Args:
        filename: Path to a RAW file.

    Returns:
        Dictionary of header key-value pairs.
    """
    header_dict = {}
    with open(filename, "rb") as f:
        chunk = f.read(80)
        while f"{'END':<80}".encode() not in chunk:
            key, val = get_header_key_val(chunk.decode())
            header_dict[key] = val
            chunk = f.read(80)
    return header_dict


def get_stem(filename: PathLike) -> Path:
    """Extract the RAW stem from a RAW filename.

    Args:
        filename: Path to a specific RAW file.

    Returns:
        RAW file stem.
    """
    raw_path = Path(filename)
    return raw_path.parent / ''.join(raw_path.stem.split('.')[:-1])


def get_raw_params(input_file_stem: PathLike,
                   start_chan: int = 0) -> dict[str, Any]:
    """Return selected observing parameters from a RAW header.

    Args:
        input_file_stem: RAW file stem.
        start_chan: Index of the first coarse channel to be recorded.

    Returns:
        Dictionary of parsed RAW parameters.
    """
    header = read_header(f'{input_file_stem}.0000.raw')
    
    raw_params = {}
    raw_params['num_bits'] = int(header['NBITS'])
    raw_params['chan_bw'] = chan_bw = float(header['CHAN_BW']) * 1e6
    raw_params['ascending'] = (chan_bw > 0)
    
    num_pols = int(header['NPOL'])
    if num_pols == 4:
        num_pols = 2
    raw_params['num_pols'] = num_pols
    
    raw_params['block_size'] = int(header['BLOCSIZE'])
    raw_params['obs_length'] = float(header['SCANLEN'])
    raw_params['tbin'] = float(header['TBIN'])
    
    try:
        num_antennas = int(header['NANTS'])
    except KeyError:
        num_antennas = 1
    raw_params['num_antennas'] = num_antennas
    
    raw_params['num_chans'] = num_chans = int(header['OBSNCHAN']) // num_antennas
    raw_params['center_freq'] = center_freq = float(header['OBSFREQ']) * 1e6
    raw_params['fch1'] = center_freq - (start_chan + (num_chans - 1) / 2) * chan_bw
    
    return raw_params


def get_blocks_in_file(filename: PathLike) -> int:
    """Return the number of data blocks in a RAW file.

    Args:
        filename: Path to a RAW file.

    Returns:
        Number of data blocks in the file.
    """
    
    header = read_header(filename)
    with open(filename, "rb") as f:
        count = 0
        block_read_size = int(512 * np.ceil((80 * (len(header) + 1)) / 512)) + int(header['BLOCSIZE'])
        while f.read(block_read_size):
#             chunk = f.read(block_read_size)
#             if len(chunk) == 0:
#                 break
#             print(len(chunk))
            count += 1
    return count


def get_blocks_per_file(input_file_stem: PathLike) -> int:
    """Return blocks in the first file matching a RAW stem.

    Args:
        input_file_stem: RAW file stem.

    Returns:
        Number of data blocks in the first file.
    """
    first_file = f'{input_file_stem}.0000.raw'
    return get_blocks_in_file(first_file)


def get_total_blocks(input_file_stem: PathLike) -> int:
    """Return the total number of blocks across all files in a RAW stem.

    Args:
        input_file_stem: RAW file stem.

    Returns:
        Total number of data blocks.
    """
    filenames = glob.glob(f'{input_file_stem}.????.raw')
    blocks_per_file = get_blocks_per_file(input_file_stem)
    if len(filenames) == 1:
        return blocks_per_file
    else:
        blocks_in_last_file = get_blocks_in_file(filenames[-1])
        return blocks_per_file * (len(filenames) - 1) + blocks_in_last_file


def get_dists(filename: PathLike, show: bool = True) -> None:
    """Plot and print component distributions from the first RAW block.

    Args:
        filename: Path to a RAW file.
        show: Whether to display the histogram plots.
    """
    header = read_header(filename)
    with open(filename, "rb") as f:
        header_size = int(512 * np.ceil((80 * (len(header) + 1)) / 512))
        f.read(header_size)
        
        block_size = int(header['BLOCSIZE'])
        chunk = f.read(block_size)

        try:
            num_antennas = int(header['NANTS'])
        except KeyError:
            num_antennas = 1
        num_chans = int(header['OBSNCHAN']) // num_antennas
        
        rawbuffer = np.frombuffer(chunk, dtype=np.int8).reshape((num_chans, -1))
        
        num_pols = int(header['NPOL'])
        if num_pols == 4:
            num_pols = 2
            
        for pol in range(num_pols):
            for comp in range(2):
                data = rawbuffer[:, comp+2*pol::2*num_pols]
                plt.hist(data.flatten(), bins=2**8)
                if show:
                    plt.show()
                fwhm_factor = 2 * np.sqrt(2 * np.log(2))
                mean = np.mean(data)
                std = np.std(data)
                print(f'Pol {pol}, comp {comp}: mean {mean}, std {std}, fwhm {std*fwhm_factor}')
