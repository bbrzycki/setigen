import sys
import os.path
try:
    import cupy as xp
except ImportError:
    import numpy as xp
import numpy as np

from tqdm import tqdm, trange

import time

from setigen import unit_utils
from . import sigproc
from . import antenna


def format_header_line(key, value):
    if isinstance(value, str):
        value = f"'{value: <8}'"
        line = f"{key:<8}= {value:<20}"
    else:
        if key == 'TBIN':
            value = f"{value:.14E}"
        line = f"{key:<8}= {value:>20}"
    line = f"{line:<80}"
    return line


def get_total_obs_num_samples(obs_length=None, 
                              num_blocks=None, 
                              num_antennas=1,
                              sample_rate=3e9,
                              block_size=134217728,
                              num_bits=8,
                              num_pols=2,
                              num_branches=1024,
                              num_chans=64, 
                              length_mode='obs_length'):
    """
    length_mode can be 'obs_length' or 'num_blocks'
    """
    tbin = num_branches / sample_rate
    chan_bw = 1 / tbin
    bytes_per_sample = 2 * num_pols * num_bits / 8
    if length_mode == 'obs_length':
        if obs_length is None:
            raise ValueError("Value not given for 'obs_length'.")
        num_blocks = int(obs_length * chan_bw * num_antennas * num_chans * bytes_per_sample / block_size)
    elif length_mode == 'num_blocks':
        if num_blocks is None:
            raise ValueError("Value not given for 'num_blocks'.")
        pass
    else:
        raise ValueError("Invalid option given for 'length_mode'.")
    return num_blocks * block_size / (num_antennas * num_chans * bytes_per_sample) * num_branches
                  
    
class RawVoltageBackend(object):
    def __init__(self,
                 antenna_source,
                 block_size=134217728,
                 blocks_per_file=128,
                 digitizer=sigproc.RealQuantizer(),
                 filterbank=sigproc.PolyphaseFilterbank(),
                 requantizer=sigproc.ComplexQuantizer()):
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
            
        self.block_size = block_size
        self.blocks_per_file = blocks_per_file
        
        self.digitizer = digitizer
        
        self.filterbank = filterbank
        if self.is_antenna_array:
            self.filterbank.cache = [[None, None] for a in range(self.num_antennas)]
        self.num_taps = self.filterbank.num_taps
        self.num_branches = self.filterbank.num_branches
        self.tbin = self.num_branches / self.sample_rate
        self.chan_bw = 1 / self.tbin
        if not self.antenna_source.ascending:
            self.chan_bw = -self.chan_bw
        
        self.requantizer = requantizer
        self.num_bits = self.requantizer.num_bits
        self.num_bytes = self.num_bits // 8
        self.bytes_per_sample = 2 * self.num_pols * self.num_bits / 8
        self.total_obs_num_samples = None
    
    def _make_header(self, f, header_dict={}):
        my_path = os.path.abspath(os.path.dirname(__file__))
        path = os.path.join(my_path, 'header_template.txt')
        with open('header_template.txt', 'r') as t:
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
                     requantize=True):
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
    
#         mempool = xp.get_default_memory_pool()
        
        with tqdm(total=self.num_antennas*self.num_pols*num_subblocks, leave=False) as pbar:
            pbar.set_description('Subblocks')

            for subblock in range(num_subblocks):
                tqdm.write(f'Creating subblock {subblock}...')

                # Change num windows at the end if num_subblocks doesn't go in evenly
                if T % subblock_T != 0 and subblock == num_subblocks - 1:
                    W = int((T % subblock_T) / self.num_taps) + 1

                if self.antenna_source.start_obs:
                    num_samples = self.num_branches * self.num_taps * W
                else:
                    num_samples = self.num_branches * self.num_taps * (W - 1)
                    
                # Calculate the real voltage samples from each antenna
                antennas_v = self.antenna_source.get_samples(num_samples, 
                                                             total_obs_num_samples=self.total_obs_num_samples)

                for antenna in range(self.num_antennas):
                    if self.is_antenna_array:
                        tqdm.write(f'Creating antenna #{antenna}...')
                
                    for pol in range(self.num_pols):
                        v = antennas_v[antenna][pol]

                        if digitize:
                            v = self.digitizer.quantize(v)

                        v = self.filterbank.channelize(v, pol=pol, antenna=antenna)
                        # Drop out last coarse channel
                        v = v[:, :-1][:, start_chan:start_chan+num_chans]

                        if requantize:
                            v = self.requantizer.quantize(v)

                        # Convert to numpy array if using cupy
                        try:
                            R = xp.asnumpy(xp.real(v).T)  
                            I = xp.asnumpy(xp.imag(v).T)  
                        except AttributeError:
                            R = xp.real(v).T
                            I = xp.imag(v).T
                            
                        c_idx = antenna * num_chans + np.arange(0, num_chans)
                        if self.num_bits == 8:
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
        
    def record(self, 
               raw_file_stem,
               obs_length=None, 
               num_blocks=None,
               start_chan=0, 
               num_chans=64, 
               num_subblocks=1,
               length_mode='obs_length',
               header_dict={},
               digitize=True):
        """
        length_mode can be 'obs_length' or 'num_blocks'
        """
        if length_mode == 'obs_length':
            if obs_length is None:
                raise ValueError("Value not given for 'obs_length'.")
            self.num_blocks = int(obs_length * abs(self.chan_bw) * self.num_antennas * num_chans * self.bytes_per_sample / self.block_size)
        elif length_mode == 'num_blocks':
            if num_blocks is None:
                raise ValueError("Value not given for 'num_blocks'.")
            self.num_blocks = num_blocks
        else:
            raise ValueError("Invalid option given for 'length_mode'.")
        self.obs_length = self.num_blocks * self.block_size / (self.num_antennas * num_chans * self.bytes_per_sample) * self.tbin
        self.total_obs_num_samples = self.obs_length // self.tbin * self.num_branches
            
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
                    center_freq = (start_chan + (num_chans - 1) / 2) * self.chan_bw
                    center_freq += self.antenna_source.fch1
                        
                    for j in range(blocks_to_write):
                        tqdm.write(f'Creating block {j}...')
                        # Set additional header values according to which band is recorded
                        header_dict['OBSNCHAN'] = num_chans * self.num_antennas
                        header_dict['OBSFREQ'] = center_freq * 1e-6
                        header_dict['OBSBW'] = self.chan_bw * num_chans * 1e-6
                        self._make_header(f, header_dict)

                        v = self.collect_data(start_chan=start_chan, 
                                              num_chans=num_chans,
                                              num_subblocks=num_subblocks,
                                              digitize=digitize, 
                                              requantize=True)

                        f.write(xp.array(v, dtype=xp.int8).tobytes())
                        tqdm.write(f'File {i}, block {j} recorded!')
                        pbar.update(1)
