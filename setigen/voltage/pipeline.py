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


class RawVoltagePipeline(object):
    def __init__(self,
                 antenna,
                 block_size=134217728,
                 blocks_per_file=128,
                 digitizer=sigproc.RealQuantizer(),
                 filterbank=sigproc.PolyphaseFilterbank(),
                 requantizer=sigproc.ComplexQuantizer()):
        self.antenna = antenna
        self.sample_rate = self.antenna.sample_rate
        self.num_pols = self.antenna.num_pols
        if self.num_pols == 2:
            self.streams = [self.antenna.x, self.antenna.y]
        else:
            self.streams = [self.antenna.x]
            
        self.block_size = block_size
        self.blocks_per_file = blocks_per_file
        
        self.digitizer = digitizer
        
        self.filterbank = filterbank
        self.num_taps = self.filterbank.num_taps
        self.num_branches = self.filterbank.num_branches
        
        self.requantizer = requantizer
        self.num_bits = self.requantizer.num_bits
        self.num_bytes = self.num_bits // 8
        
    def _format_header_line(self, key, value):
        if isinstance(value, str):
            value = f"'{value: <8}'"
            line = f"{key:<8}= {value:<20}"
        else:
            if key == 'TBIN':
                value = f"{value:.14E}"
            line = f"{key:<8}= {value:>20}"
        line = f"{line:<80}"
        return line
    
    def _make_header(self, f, header_dict={}):
        my_path = os.path.abspath(os.path.dirname(__file__))
        path = os.path.join(my_path, 'header_template.txt')
        with open('header_template.txt', 'r') as t:
            template_lines = t.readlines()
            
        # Set header values determined by pipeline parameters
        header_dict['TELESCOP'] = 'SETIGEN'
        header_dict['OBSERVER'] = 'SETIGEN'
        header_dict['SRC_NAME'] = 'SYNTHETIC'
        
        header_dict['NBITS'] = self.num_bits
        header_dict['CHAN_BW'] = self.sample_rate / self.num_branches * 1e-6
        if self.num_pols == 2:
            header_dict['NPOL'] = 4
        else:
            header_dict['NPOL'] = self.num_pols
        header_dict['BLOCSIZE'] = self.block_size
        header_dict['SCANLEN'] = self.obs_length
        header_dict['TBIN'] = self.num_branches / self.sample_rate
        
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
        
    def set_obs_length(self, obs_length):
        self.obs_length = obs_length
        self.num_blocks = int(obs_length * self.sample_rate * 2 * self.num_pols * self.num_bits / 8 / self.block_size)
        
    def set_num_blocks(self, num_blocks):
        self.num_blocks = num_blocks
        self.obs_length = num_blocks * self.block_size / (self.sample_rate * 2 * self.num_pols * self.num_bits / 8)
        
    def collect_data(self,
                     start_chan,
                     num_chans,
                     num_subblocks=1,
                     digitize=True,
                     requantize=True):
        final_voltages = np.empty((num_chans, int(self.block_size / num_chans)))
    
        # Make sure that block_size is appropriate
        assert self.block_size % int(2 * self.num_pols * num_chans * self.num_taps * self.num_bits / 8) == 0
        T = int(self.block_size / (2 * self.num_pols * num_chans * self.num_bits / 8))

        W = int(xp.ceil(T / self.num_taps / num_subblocks)) + 1
        subblock_T = self.num_taps * (W - 1)

        # Change num_subblocks if necessary
        num_subblocks = int(xp.ceil(T / subblock_T))
        subblock_t_len = int(subblock_T * 2 * self.num_pols * self.num_bits / 8)
    
        mempool = xp.get_default_memory_pool()
    
        with tqdm(total=self.num_pols*num_subblocks, leave=False) as pbar:
            pbar.set_description('Subblocks')
            
            for pol, stream in enumerate(self.streams):
                tqdm.write(f'Creating polarization #{pol+1}...')

                for subblock in range(num_subblocks):
                    tqdm.write(f'Creating subblock {subblock}...')

#                     bytenum=subblock_T * 2 * self.num_pols * 8 * self.num_branches
#                     print(bytenum)
#                     print('used',mempool.used_bytes())              # 0
#                     print('total',mempool.total_bytes())
#                     print(mempool.total_bytes()/bytenum)

                    # Change num windows at the end if num_subblocks doesn't go in evenly
                    if T % subblock_T != 0 and subblock == num_subblocks - 1:
                        W = int((T % subblock_T) / self.num_taps) + 1
    #                     print(T, subblock_T, W-1)

                    if stream.start_obs:
                        num_samples = self.num_branches * self.num_taps * W
                    else:
                        num_samples = self.num_branches * self.num_taps * (W - 1)

    #                 start = time.time()

                    v = stream.get_samples(num_samples=num_samples)

    #                 print('samp',time.time() - start)
    #                 start = time.time()

                    if digitize:
                        v = self.digitizer.quantize(v)

    #                 print('quan',time.time() - start)
    #                 start = time.time()

                    # Drop out last coarse channel
                    v = self.filterbank.channelize(v, pol=pol)[:, :-1][:, start_chan:start_chan+num_chans]

    #                 print('chan',time.time() - start)
    #                 start = time.time()

                    if requantize:
                        v = self.requantizer.quantize(v)

    #                 print('requan',time.time() - start)
    #                 start = time.time()

                    if self.num_bits == 8:
                        if T % subblock_T != 0 and subblock == num_subblocks - 1:
                            # Uses adjusted W
                            idx = subblock * subblock_t_len + 2 * pol + np.arange(0, self.num_taps * (W - 1) * 2 * self.num_pols, 2 * self.num_pols)
                        else:
                            idx = subblock * subblock_t_len + 2 * pol + np.arange(0, subblock_t_len, 2 * self.num_pols)
                        final_voltages[:, idx] = xp.asnumpy(xp.real(v).T)
                        final_voltages[:, idx+1] = xp.asnumpy(xp.imag(v).T)
                    elif self.num_bits == 4:
                        if T % subblock_T != 0 and subblock == num_subblocks - 1:
                            # Uses adjusted W
                            idx = subblock * subblock_t_len + pol + np.arange(0, self.num_taps * (W - 1) * self.num_pols, self.num_pols)
                        else:
                            idx = subblock * subblock_t_len + pol + np.arange(0, subblock_t_len, self.num_pols)

                        R = xp.asnumpy(xp.real(v).T)  
                        I = xp.asnumpy(xp.imag(v).T)  
                        # Translate 4 bit complex voltages to an 8 bit equivalent representation
                        I[I < 0] += 16
                        final_voltages[:, idx] = R * 16 + I
                    else:
                        sys.exit(f'{self.num_bits} bits not supported...')

    #                 print('trunc',time.time() - start)
    #                 start = time.time()
    
                    pbar.update(1)
            
        return final_voltages    
        
    def record(self, 
               raw_file_stem,
               obs_length=None, 
               num_blocks=None,
               start_chan=None, 
               num_chans=None, 
               num_subblocks=1,
               length_mode='obs_length',
               header_dict={},
               digitize=True):
        """
        length_mode can be 'obs_length' or 'num_blocks'
        """
        if length_mode == 'obs_length':
            self.set_obs_length(obs_length)
        elif length_mode == 'num_blocks':
            self.set_num_blocks(num_blocks)
        else:
            raise ValueError("Invalid option given for 'length_mode'.")
            
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

                    # This is 1 higher if we exclude the DC bin
                    center_freq = (1 + start_chan + (num_chans - 1) / 2) * self.sample_rate / self.num_branches

                    for j in range(blocks_to_write):
                        tqdm.write(f'Creating block {j}...')
                        # Set additional header values according to which band is recorded
                        header_dict['OBSNCHAN'] = num_chans
                        header_dict['OBSFREQ'] = center_freq * 1e-6
                        header_dict['OBSBW'] = self.sample_rate / self.num_branches * num_chans * 1e-6
                        self._make_header(f, header_dict)

                        v = self.collect_data(start_chan=start_chan, 
                                              num_chans=num_chans,
                                              num_subblocks=num_subblocks,
                                              digitize=digitize, 
                                              requantize=True)

                        f.write(xp.array(v, dtype=xp.int8).tobytes())
                        tqdm.write(f'Block {j} recorded')
                        pbar.update(1)


                        
class MultiAntennaVoltagePipeline(object):
    def __init__(self,
                 antenna_array,
                 block_size=134217728,
                 blocks_per_file=128,
                 digitizer=sigproc.RealQuantizer(),
                 filterbank=sigproc.PolyphaseFilterbank(),
                 requantizer=sigproc.ComplexQuantizer()):
        self.antenna_array = antenna_array
        self.num_antennas = self.antenna_array.num_antennas
        self.sample_rate = self.antenna_array.sample_rate
        self.num_pols = self.antenna_array.num_pols
            
        self.block_size = block_size
        self.blocks_per_file = blocks_per_file
        
        self.digitizer = digitizer
        
        self.filterbank = filterbank
        self.filterbank.cache = [[None, None] for a in range(self.num_antennas)]
        self.num_taps = self.filterbank.num_taps
        self.num_branches = self.filterbank.num_branches
        
        self.requantizer = requantizer
        self.num_bits = self.requantizer.num_bits
        self.num_bytes = self.num_bits // 8
        
    def _format_header_line(self, key, value):
        if isinstance(value, str):
            value = f"'{value: <8}'"
            line = f"{key:<8}= {value:<20}"
        else:
            if key == 'TBIN':
                value = f"{value:.14E}"
            line = f"{key:<8}= {value:>20}"
        line = f"{line:<80}"
        return line
    
    def _make_header(self, f, header_dict={}):
        my_path = os.path.abspath(os.path.dirname(__file__))
        path = os.path.join(my_path, 'header_template.txt')
        with open('header_template.txt', 'r') as t:
            template_lines = t.readlines()
            
        # Set header values determined by pipeline parameters
        header_dict['TELESCOP'] = 'SETIGEN'
        header_dict['OBSERVER'] = 'SETIGEN'
        header_dict['SRC_NAME'] = 'SYNTHETIC'
        
        header_dict['NBITS'] = self.num_bits
        header_dict['CHAN_BW'] = self.sample_rate / self.num_branches * 1e-6
        if self.num_pols == 2:
            header_dict['NPOL'] = 4
        else:
            header_dict['NPOL'] = self.num_pols
        header_dict['BLOCSIZE'] = self.block_size
        header_dict['SCANLEN'] = self.obs_length
        header_dict['TBIN'] = self.num_branches / self.sample_rate
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
        
    def set_obs_length(self, obs_length):
        self.obs_length = obs_length
        self.num_blocks = int(obs_length * self.sample_rate * 2 * self.num_pols * self.num_bits / 8 / self.block_size * self.num_antennas)
        
    def set_num_blocks(self, num_blocks):
        self.num_blocks = num_blocks
        self.obs_length = num_blocks * self.block_size / (self.sample_rate * 2 * self.num_pols * self.num_bits / 8 * self.num_antennas)
        
    def collect_data(self,
                     start_chan,
                     num_chans,
                     num_subblocks=1,
                     digitize=True,
                     requantize=True):
        obsnchan = num_chans * self.num_antennas
        final_voltages = np.empty((obsnchan, int(self.block_size / obsnchan)))
    
        # Make sure that block_size is appropriate
        assert self.block_size % int(2 * self.num_pols * num_chans * self.num_taps * self.num_bits / 8 * self.num_antennas) == 0
        T = int(self.block_size / (2 * self.num_pols * num_chans * self.num_bits / 8 * self.num_antennas))

        W = int(xp.ceil(T / self.num_taps / num_subblocks)) + 1
        subblock_T = self.num_taps * (W - 1)

        # Change num_subblocks if necessary
        num_subblocks = int(xp.ceil(T / subblock_T))
        subblock_t_len = int(subblock_T * 2 * self.num_pols * self.num_bits / 8)
    
        mempool = xp.get_default_memory_pool()
        
        with tqdm(total=self.num_antennas*self.num_pols*num_subblocks, leave=False) as pbar:
            pbar.set_description('Antenna subblocks')

            for subblock in range(num_subblocks):
                tqdm.write(f'Creating subblock {subblock}...')

                # Change num windows at the end if num_subblocks doesn't go in evenly
                if T % subblock_T != 0 and subblock == num_subblocks - 1:
                    W = int((T % subblock_T) / self.num_taps) + 1

                if self.antenna_array.start_obs:
                    num_samples = self.num_branches * self.num_taps * W
                else:
                    num_samples = self.num_branches * self.num_taps * (W - 1)
                antennas_v = self.antenna_array.get_samples(num_samples)

                for antenna in range(self.num_antennas):
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

                        c_idx = antenna * num_chans + np.arange(0, num_chans)
                        if self.num_bits == 8:
                            if T % subblock_T != 0 and subblock == num_subblocks - 1:
                                # Uses adjusted W
                                t_idx = subblock * subblock_t_len + 2 * pol + np.arange(0, self.num_taps * (W - 1) * 2 * self.num_pols, 2 * self.num_pols)
                            else:
                                t_idx = subblock * subblock_t_len + 2 * pol + np.arange(0, subblock_t_len, 2 * self.num_pols)
                            final_voltages[c_idx[:, np.newaxis], t_idx[np.newaxis, :]] = xp.asnumpy(xp.real(v).T)
                            final_voltages[c_idx[:, np.newaxis], (t_idx+1)[np.newaxis, :]] = xp.asnumpy(xp.imag(v).T)
                        elif self.num_bits == 4:
                            if T % subblock_T != 0 and subblock == num_subblocks - 1:
                                # Uses adjusted W
                                t_idx = subblock * subblock_t_len + pol + np.arange(0, self.num_taps * (W - 1) * self.num_pols, self.num_pols)
                            else:
                                t_idx = subblock * subblock_t_len + pol + np.arange(0, subblock_t_len, self.num_pols)

                            R = xp.asnumpy(xp.real(v).T)  
                            I = xp.asnumpy(xp.imag(v).T)  
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
               start_chan=None, 
               num_chans=None, 
               num_subblocks=1,
               length_mode='obs_length',
               header_dict={},
               digitize=True):
        """
        length_mode can be 'obs_length' or 'num_blocks'
        """
        if length_mode == 'obs_length':
            self.set_obs_length(obs_length)
        elif length_mode == 'num_blocks':
            self.set_num_blocks(num_blocks)
        else:
            raise ValueError("Invalid option given for 'length_mode'.")
            
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

                    # This is 1 higher if we exclude the DC bin
                    center_freq = (1 + start_chan + (num_chans - 1) / 2) * self.sample_rate / self.num_branches

                    for j in range(blocks_to_write):
                        tqdm.write(f'Creating block {j}...')
                        # Set additional header values according to which band is recorded
                        header_dict['OBSNCHAN'] = num_chans * self.num_antennas
                        header_dict['OBSFREQ'] = center_freq * 1e-6
                        header_dict['OBSBW'] = self.sample_rate / self.num_branches * num_chans * 1e-6
                        self._make_header(f, header_dict)

                        v = self.collect_data(start_chan=start_chan, 
                                              num_chans=num_chans,
                                              num_subblocks=num_subblocks,
                                              digitize=digitize, 
                                              requantize=True)

                        f.write(xp.array(v, dtype=xp.int8).tobytes())
                        tqdm.write(f'Block {j} recorded')
                        pbar.update(1)
