import numpy as np
import glob


def format_header_line(key, value, as_strings=False):
    """
    Format key, value pair as an 80 character RAW header line.
    
    Parameters
    ----------
    key : str
        Header key
    value : str or int or float
        Header value
    as_strings : bool
        If values are already formatted strings, pass True
        
    Returns
    -------
    line : str
        Formatted line
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


def get_header_key_val(header_line):
    """
    Split header_line into key, value pair.
    
    Parameters
    ----------
    header_line : str
        Formatted header line
        
    Returns
    -------
    key : str
        Header key
    value : str
        Header value (as string)
    """
    key = header_line[:8].strip()
    value = header_line[9:].strip()#.strip("''")
    return key, value


def read_header(filename):
    """
    Return header dictionary, read from a GUPPI RAW file.
    
    Parameters
    ----------
    filename : str
        Path to RAW file
        
    Returns
    -------
    header_dict : dict
        Dictionary of header key, value pairs
    """
    header_dict = {}
    with open(filename, "rb") as f:
        chunk = f.read(80)
        while f"{'END':<80}".encode() not in chunk:
            key, val = get_header_key_val(chunk.decode())
            header_dict[key] = val
            chunk = f.read(80)
    return header_dict


def get_raw_params(input_file_stem,
                   start_chan=0):
    """
    Return dictionary with parameters from RAW file's header.
    
    Parameters
    ----------
    input_file_stem : str
        Path to RAW file stem (prefix)
    start_chan : int, optional
        Index of first coarse channel to be recorded
        
    Returns
    -------
    raw_params : dict
        Dictionary with header parameters
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


def get_blocks_in_file(filename):
    """
    Return number of blocks within a RAW file.
    
    Parameters
    ----------
    filename : str
        Path to RAW file
        
    Returns
    -------
    count : int
        Number of data blocks
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


def get_blocks_per_file(input_file_stem):
    """
    Return blocks in first file matching the filename stem.
    
    Parameters
    ----------
    input_file_stem : str
        Path to RAW file stem (prefix)
        
    Returns
    -------
    count : int
        Number of data blocks
    """
    first_file = f'{input_file_stem}.0000.raw'
    return get_blocks_in_file(first_file)


def get_total_blocks(input_file_stem):
    """
    Return total number of blocks in data.
    
    Parameters
    ----------
    input_file_stem : str
        Path to RAW file stem (prefix)
        
    Returns
    -------
    num_blocks : int
        Number of data blocks
    """
    filenames = glob.glob(f'{input_file_stem}.????.raw')
    blocks_per_file = get_blocks_per_file(input_file_stem)
    if len(filenames) == 1:
        return blocks_per_file
    else:
        blocks_in_last_file = get_blocks_in_file(filenames[-1])
        return blocks_per_file * (len(filenames) - 1) + blocks_in_last_file


def get_dists(filename):
    header = read_header(filename)
    with open(filename, "rb") as f:
        i = 0
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
                plt.show()
                fwhm_factor = 2 * np.sqrt(2 * np.log(2))
                mean = np.mean(data)
                std = np.std(data)
                print(f'Pol {pol}, comp {comp}: mean {mean}, std {std}, fwhm {std*fwhm_factor}')