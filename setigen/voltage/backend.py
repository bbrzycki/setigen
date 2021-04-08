import sys
import os

GPU_FLAG = os.getenv('SETIGEN_ENABLE_GPU', '0')
if GPU_FLAG == '1':
    try:
        import cupy as xp
    except ImportError:
        import numpy as xp
else:
    import numpy as xp
import numpy as np

from tqdm import tqdm

import time

from setigen import unit_utils
from . import polyphase_filterbank
from . import quantization
from . import antenna


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


class RawVoltageBackend(object):
    """
    Central class that wraps around antenna sources and backend elements to facilitate the
    creation of GUPPI RAW voltage files from synthetic real voltages.
    """
    def __init__(self,
                 antenna_source,
                 digitizer=quantization.RealQuantizer(),
                 filterbank=polyphase_filterbank.PolyphaseFilterbank(),
                 requantizer=quantization.ComplexQuantizer(),
                 start_chan=0,
                 num_chans=64,
                 block_size=134217728,
                 blocks_per_file=128,
                 num_subblocks=32):
        """
        Initialize a RawVoltageBackend object, with an input antenna source (either Antenna or 
        MultiAntennaArray), and backend elements (digitizer, filterbank, requantizer). Also, details 
        behind the RAW file format and recording are specified on initialization, such as which
        coarse channels are saved and the size of recording blocks.

        Parameters
        ----------
        antenna_source : Antenna or MultiAntennaArray
            Antenna or MultiAntennaArray, from which real voltage data is created
        digitizer : RealQuantizer or ComplexQuantizer, optional
            Quantizer used to digitize input voltages
        filterbank : PolyphaseFilterbank, optional
            Polyphase filterbank object used to channelize voltages
        requantizer : ComplexQuantizer, optional
            Quantizer used on complex channelized voltages
        start_chan : int, optional
            Index of first coarse channel to be recorded
        num_chans : int, optional
            Number of coarse channels to be recorded
        block_size : int, optional
            Recording block size, in bytes
        blocks_per_file : int, optional
            Number of blocks to be saved per RAW file
        num_subblocks : int, optional
            Number of partitions per block, used for computation. If `num_subblocks`=1, one block's worth
            of data will be passed through the pipeline and recorded at once. Use this parameter to reduce 
            memory load, especially when using GPU acceleration.
        """
        self.antenna_source = antenna_source
        if isinstance(antenna_source, antenna.Antenna):
            self.num_antennas = 1
            self.is_antenna_array = False
        elif isinstance(antenna_source, antenna.MultiAntennaArray):
            self.num_antennas = self.antenna_source.num_antennas
            self.is_antenna_array = True
        else:
            raise ValueError("Invalid type provided for 'antenna_source'.")
        self.sample_rate = self.antenna_source.sample_rate
        self.num_pols = self.antenna_source.num_pols
        self.fch1 = self.antenna_source.fch1
        self.ascending = self.antenna_source.ascending
            
        self.start_chan = start_chan
        self.num_chans = num_chans
        self.block_size = block_size
        self.blocks_per_file = blocks_per_file
        self.num_subblocks = num_subblocks
        
        self.digitizer = digitizer
        if self.is_antenna_array:
            if isinstance(self.digitizer, quantization.RealQuantizer):
                self.digitizer.stats_cache = [[[None, None]] * self.num_pols] * self.num_antennas
                self.digitizer.stats_calc_indices = [[0] * self.num_pols] * self.num_antennas
            elif isinstance(self.digitizer, quantization.ComplexQuantizer):
                for sub_quantizer in [self.digitizer.quantizer_r, self.digitizer.quantizer_i]:
                    sub_quantizer.stats_cache = [[[None, None]] * self.num_pols] * self.num_antennas
                    sub_quantizer.stats_calc_indices = [[0] * self.num_pols] * self.num_antennas
        
        self.filterbank = filterbank
        assert isinstance(self.filterbank, polyphase_filterbank.PolyphaseFilterbank)
        if self.is_antenna_array:
            self.filterbank.cache = [[None] * self.num_pols] * self.num_antennas
        self.num_taps = self.filterbank.num_taps
        self.num_branches = self.filterbank.num_branches
        assert self.start_chan + self.num_chans <= self.num_branches // 2
        
        self.tbin = self.num_branches / self.sample_rate
        self.chan_bw = 1 / self.tbin
        if not self.ascending:
            self.chan_bw = -self.chan_bw
        
        self.requantizer = requantizer
        assert isinstance(self.requantizer, quantization.ComplexQuantizer)
        if self.is_antenna_array:
            for sub_quantizer in [self.requantizer.quantizer_r, self.requantizer.quantizer_i]:
                sub_quantizer.stats_cache = [[[None, None]] * self.num_pols] * self.num_antennas
                sub_quantizer.stats_calc_indices = [[0] * self.num_pols] * self.num_antennas
        self.num_bits = self.requantizer.num_bits
        self.num_bytes = self.num_bits // 8
        self.bytes_per_sample = 2 * self.num_pols * self.num_bits / 8
        self.total_obs_num_samples = None
        
        self.time_per_block = self.block_size / (self.num_antennas * self.num_chans * self.bytes_per_sample) * self.tbin
    
        # Make sure that block_size is appropriate
        assert self.block_size % int(self.num_antennas * self.num_chans * self.num_taps * self.bytes_per_sample) == 0
        
        self.sample_stage_t = 0
        self.digitizer_stage_t = 0
        self.filterbank_stage_t = 0
        self.requantizer_stage_t = 0
        
    
    def _make_header(self, f, header_dict={}):
        """
        Write all header lines out to file as bytes.
        
        Parameters
        ----------
        f : file handle
            File handle of open RAW file
        header_dict : dict, optional
            Dictionary of header values to set. Use to overwrite non-essential header values or add custom ones.
        """
        my_path = os.path.abspath(os.path.dirname(__file__))
        path = os.path.join(my_path, 'header_template.txt')
        with open(path, 'r') as t:
            template_lines = t.readlines()
            
        # Set header values determined by pipeline parameters
        if 'TELESCOP' not in header_dict:
            header_dict['TELESCOP'] = 'SETIGEN'
        if 'OBSERVER' not in header_dict:
            header_dict['OBSERVER'] = 'SETIGEN'
        if 'SRC_NAME' not in header_dict:
            header_dict['SRC_NAME'] = 'SYNTHETIC'
        
        # Should not be able to manually change these header values
        header_dict['NBITS'] = self.num_bits
        header_dict['CHAN_BW'] = self.chan_bw * 1e-6
        if self.num_pols == 2:
            header_dict['NPOL'] = 4
        else:
            header_dict['NPOL'] = self.num_pols
        header_dict['BLOCSIZE'] = self.block_size
        header_dict['SCANLEN'] = self.obs_length
        header_dict['TBIN'] = self.tbin
        if self.is_antenna_array:
            header_dict['NANTS'] = self.num_antennas
        
        header_lines = []
        used_keys = set()
        for i in range(len(template_lines) - 1):
            key = template_lines[i][:8].strip()
            used_keys.add(key)
            if key in header_dict:
                header_lines.append(format_header_line(key, header_dict[key]))
            else:
                # Cut newline character
                header_lines.append(template_lines[i][:-1])
        for key in header_dict.keys() - used_keys:
            header_lines.append(format_header_line(key, header_dict[key]))
        header_lines.append(f"{'END':<80}")
        
        # Write each line with space and zero padding
        for line in header_lines:
            f.write(f"{line:<80}".encode())
        f.write(bytearray(512 - (80 * len(header_lines) % 512)))   
        
    def collect_data(self,
                     start_chan,
                     num_chans,
                     num_subblocks=1,
                     digitize=True,
                     requantize=True,
                     verbose=True):
        """
        General function to actually collect data from the antenna source and return coarsely channelized complex
        voltages.
        
        Parameters
        ----------
        start_chan : int
            Index of first coarse channel to be saved
        num_chans : int
            Number of coarse channels to be saved
        num_subblocks : int, optional
            Number of partitions per block, used for computation. If `num_subblocks`=1, one block's worth
            of data will be passed through the pipeline and recorded at once. Use this parameter to reduce 
            memory load, especially when using GPU acceleration.
        digitize : bool, optional
            Whether to quantize input voltages before the PFB
        requantize : bool, optional
            Whether to quantize output complex voltages after the PFB
        verbose : bool, optional
            Control whether tqdm prints progress messages 
            
        Returns
        -------
        final_voltages : array
            Complex voltages formatted according to GUPPI RAW specifications; array of shape 
            (num_chans * num_antennas, block_size / (num_chans * num_antennas))
        """
        obsnchan = num_chans * self.num_antennas
        final_voltages = np.empty((obsnchan, int(self.block_size / obsnchan)))
    
        # Make sure that block_size is appropriate
        assert self.block_size % int(self.num_antennas * num_chans * self.num_taps * self.bytes_per_sample) == 0
        T = int(self.block_size / (self.num_antennas * num_chans * self.bytes_per_sample))

        W = int(xp.ceil(T / self.num_taps / num_subblocks)) + 1
        subblock_T = self.num_taps * (W - 1)

        # Change num_subblocks if necessary
        num_subblocks = int(xp.ceil(T / subblock_T))
        subblock_t_len = int(subblock_T * self.bytes_per_sample)
        
        with tqdm(total=self.num_antennas*self.num_pols*num_subblocks, leave=False) as pbar:
            pbar.set_description('Subblocks')

            for subblock in range(num_subblocks):
                if verbose:
                    tqdm.write(f'Creating subblock {subblock}...')

                # Change num windows at the end if num_subblocks doesn't go in evenly
                if T % subblock_T != 0 and subblock == num_subblocks - 1:
                    W = int((T % subblock_T) / self.num_taps) + 1

                if self.antenna_source.start_obs:
                    num_samples = self.num_branches * self.num_taps * W
                else:
                    num_samples = self.num_branches * self.num_taps * (W - 1)
                    
                # Calculate the real voltage samples from each antenna
                t = time.time()
                antennas_v = self.antenna_source.get_samples(num_samples)
                self.sample_stage_t += time.time() - t

                for antenna in range(self.num_antennas):
                    if verbose and self.is_antenna_array:
                        tqdm.write(f'Creating antenna #{antenna}...')
                
                    for pol in range(self.num_pols):
                        v = antennas_v[antenna][pol]

                        if digitize:
                            t = time.time()
                            v = self.digitizer.quantize(v, pol=pol, antenna=antenna)
                            self.digitizer_stage_t += time.time() - t

                        t = time.time()
                        v = self.filterbank.channelize(v, pol=pol, antenna=antenna)
                        # Drop out last coarse channel
                        v = v[:, start_chan:start_chan+num_chans]
                        self.filterbank_stage_t += time.time() - t

                        if requantize:
                            t = time.time()
                            v = self.requantizer.quantize(v, pol=pol, antenna=antenna)
                            self.requantizer_stage_t += time.time() - t

                        # Convert to numpy array if using cupy
                        try:
                            R = xp.asnumpy(xp.real(v).T)  
                            I = xp.asnumpy(xp.imag(v).T)  
                        except AttributeError:
                            R = xp.real(v).T
                            I = xp.imag(v).T
                            
                        c_idx = antenna * num_chans + np.arange(0, num_chans)
                        if self.num_bits == 8 or not requantize:
                            if T % subblock_T != 0 and subblock == num_subblocks - 1:
                                # Uses adjusted W
                                t_idx = subblock * subblock_t_len + 2 * pol + np.arange(0, self.num_taps * (W - 1) * 2 * self.num_pols, 2 * self.num_pols)
                            else:
                                t_idx = subblock * subblock_t_len + 2 * pol + np.arange(0, subblock_t_len, 2 * self.num_pols)
                            final_voltages[c_idx[:, np.newaxis], t_idx[np.newaxis, :]] = R
                            final_voltages[c_idx[:, np.newaxis], (t_idx+1)[np.newaxis, :]] = I
                        elif self.num_bits == 4:
                            if T % subblock_T != 0 and subblock == num_subblocks - 1:
                                # Uses adjusted W
                                t_idx = subblock * subblock_t_len + pol + np.arange(0, self.num_taps * (W - 1) * self.num_pols, self.num_pols)
                            else:
                                t_idx = subblock * subblock_t_len + pol + np.arange(0, subblock_t_len, self.num_pols)
 
                            # Translate 4 bit complex voltages to an 8 bit equivalent representation
                            I[I < 0] += 16
                            
                            final_voltages[c_idx[:, np.newaxis], t_idx[np.newaxis, :]] = R * 16 + I
                        else:
                            sys.exit(f'{self.num_bits} bits not supported...')

                        pbar.update(1)
            
        return final_voltages    
    
    def get_num_blocks(self, obs_length):
        """
        Calculate the number of blocks required as a function of observation length, in seconds. Note that only an integer
        number of blocks will be recorded, so the actual observation length may be shorter than the `obs_length` provided.
        """
        return int(obs_length * abs(self.chan_bw) * self.num_antennas * self.num_chans * self.bytes_per_sample / self.block_size)
        
    def record(self, 
               raw_file_stem,
               obs_length=None, 
               num_blocks=None,
               length_mode='obs_length',
               header_dict={},
               digitize=True,
               verbose=True):
        """
        General function to actually collect data from the antenna source and return coarsely channelized complex
        voltages.
        
        Parameters
        ----------
        raw_file_stem : str
            Filename or path stem; the suffix will be automatically appended
        obs_length : float, optional
            Length of observation in seconds, if in `obs_length` mode
        num_blocks : int, optional
            Number of data blocks to record, if in `num_blocks` mode
        length_mode : str, optional
            Mode for specifying length of observation, either `obs_length` in seconds or `num_blocks` in data blocks
        header_dict : dict, optional
            Dictionary of header values to set. Use to overwrite non-essential header values or add custom ones.
        digitize : bool, optional
            Whether to quantize input voltages before the PFB
        verbose : bool, optional
            Control whether tqdm prints progress messages 
        """
        if length_mode == 'obs_length':
            if obs_length is None:
                raise ValueError("Value not given for 'obs_length'.")
            self.num_blocks = self.get_num_blocks(obs_length)
        elif length_mode == 'num_blocks':
            if num_blocks is None:
                raise ValueError("Value not given for 'num_blocks'.")
            self.num_blocks = num_blocks
        else:
            raise ValueError("Invalid option given for 'length_mode'.")
        self.obs_length = self.num_blocks * self.time_per_block
        self.total_obs_num_samples = int(self.obs_length / self.tbin) * self.num_branches
        
        # Mark each antenna and data stream as the start of the observation
        self.antenna_source.reset_start()
        
        # Reset filterbank cache as well
        self.filterbank.cache = [[None, None] for a in range(self.num_antennas)]
            
        # Collect data and record to disk
        num_files = int(xp.ceil(self.num_blocks / self.blocks_per_file))
        with tqdm(total=self.num_blocks) as pbar:
            pbar.set_description('Blocks')
            for i in range(num_files):
                save_fn = f'{raw_file_stem}.{i:04}.raw'
                with open(save_fn, 'wb') as f:
                    # If blocks won't fill a whole file, adjust number of blocks to write at the end
                    if i == num_files - 1 and self.num_blocks % self.blocks_per_file != 0:
                        blocks_to_write = self.num_blocks % self.blocks_per_file
                    else:
                        blocks_to_write = self.blocks_per_file

                    # self.chan_bw was already adjusted for ascending or descending frequencies 
                    center_freq = (self.start_chan + (self.num_chans - 1) / 2) * self.chan_bw
                    center_freq += self.fch1
                        
                    for j in range(blocks_to_write):
                        if verbose:
                            tqdm.write(f'Creating block {j}...')
                        # Set additional header values according to which band is recorded
                        header_dict['OBSNCHAN'] = self.num_chans * self.num_antennas
                        header_dict['OBSFREQ'] = center_freq * 1e-6
                        header_dict['OBSBW'] = self.chan_bw * self.num_chans * 1e-6
                        self._make_header(f, header_dict)

                        v = self.collect_data(start_chan=self.start_chan, 
                                              num_chans=self.num_chans,
                                              num_subblocks=self.num_subblocks,
                                              digitize=digitize, 
                                              requantize=True,
                                              verbose=verbose)

                        f.write(xp.array(v, dtype=xp.int8).tobytes())
                        if verbose:
                            tqdm.write(f'File {i}, block {j} recorded!')
                        pbar.update(1)
