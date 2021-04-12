import numpy as np


def format_header_line(key, value):
    """
    Format key, value pair as an 80 character RAW header line.
    
    Parameters
    ----------
    key : str
        Header key
    value : str or int or float
        Header value
        
    Returns
    -------
    line : str
        Formatted line
    """
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
    value = header_line[9:].strip().strip("''")
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


def get_blocks_per_file(raw_file_stem):
    """
    Return blocks in first file matching the filename stem.
    
    Parameters
    ----------
    raw_file_stem : str
        Path to RAW file stem (prefix)
        
    Returns
    -------
    count : int
        Number of data blocks
    """
    first_file = f'{raw_file_stem}.0000.raw'
    return get_blocks_in_file(first_file)


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