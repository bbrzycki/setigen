import sys
import os.path
import numpy as np

import time

from setigen import unit_utils
from . import sigproc


class RawVoltagePipeline(object):
    def __init__(self,
                 stream_x, 
                 stream_y=None,
                 block_size=134217728,
                 blocks_per_file=128,
                 digitizer=sigproc.RealQuantizer(),
                 filterbank=sigproc.PolyphaseFilterbank(),
                 requantizer=sigproc.ComplexQuantizer()):
        self.stream_x = stream_x
        self.stream_y = stream_y
        if stream_y is not None:
            self.num_pols = 2
            self.streams = [self.stream_x, self.stream_y]
        else:
            self.num_pols = 1
            self.streams = [self.stream_x]
        self.sample_rate = self.stream_x.sample_rate
        assert self.stream_x.sample_rate == self.stream_y.sample_rate
            
        self.block_size = block_size
        self.blocks_per_file = blocks_per_file
        
        self.digitizer = digitizer
        
        self.filterbank = filterbank
        self.num_taps = self.filterbank.num_taps
        self.num_branches = self.filterbank.num_branches
        
        self.requantizer = requantizer
        self.num_bits = self.requantizer.num_bits
        
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
    
    def _make_header(self, header_dict={}):
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
                header_lines.append(self._format_header_line(key, header_dict[key]))
            else:
                # Cut newline character
                header_lines.append(template_lines[i][:-1])
        for key in header_dict.keys() - used_keys:
            header_lines.append(self._format_header_line(key, header_dict[key]))
        header_lines.append(f"{'END':<80}")
                
        return header_lines
    
    def _write_header(self, header_lines, f):
        # Write each line with space and zero padding
        for line in header_lines:
            f.write(f"{line:<80}".encode())
        f.write(bytearray(512 - (80 * len(header_lines) % 512)))    
        
    def set_obs_length(self, obs_length):
        self.obs_length = obs_length
        self.num_blocks = (obs_length * self.sample_rate / self.num_taps / self.num_branches - 1) * self.num_taps * 2 * self.num_pols * self.num_branches / self.block_size
        
    def set_num_blocks(self, num_blocks):
        self.num_blocks = num_blocks
        self.obs_length = (num_blocks * self.block_size / (self.num_taps * 2 * self.num_pols * self.num_branches) + 1) * self.num_taps * self.num_branches / self.sample_rate
        
    def collect_data(self,
                     num_chans,
                     digitize=True,
                     requantize=True):
        start = time.time()
        
        W = int(self.block_size / 2 / self.num_pols / self.num_taps / num_chans) + 1
        final_voltages = np.empty((self.num_branches // 2 + 1, 2 * self.num_pols * self.num_taps * (W - 1)))
        
        print('init',time.time() - start)
        start = time.time()
            
        for pol, stream in enumerate(self.streams):
            if stream.next_t_start == 0:
                num_samples = self.num_branches * self.num_taps * W
            else:
                num_samples = self.num_branches * self.num_taps * (W - 1)

            start = time.time()
            
            v = stream.get_samples(num_samples=num_samples)
            
            print('samp',time.time() - start)
            start = time.time()
            
            if digitize:
                v = self.digitizer.quantize(v)
                
            print('quan',time.time() - start)
            start = time.time()
            
            v = self.filterbank.channelize(v, pol=pol)
            
            print('chan',time.time() - start)
            start = time.time()
            
            if requantize:
                v = self.requantizer.quantize(v)
            
            print('requan',time.time() - start)
            start = time.time()
            
            final_voltages[:, 2*pol::2*self.num_pols] = np.real(v).T
            final_voltages[:, 2*pol+1::2*self.num_pols] = np.imag(v).T
            
            print('trunc',time.time() - start)
            start = time.time()
            
        return final_voltages    
        
    def record(self, 
               raw_file_stem,
               obs_length=None, 
               num_blocks=None,
               start_chan=None, 
               num_chans=None, 
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
        num_files = int(np.ceil(self.num_blocks / self.blocks_per_file))
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
                    print(f'Creating block {j}...')
                    # Set additional header values according to which band is recorded
                    header_dict['OBSNCHAN'] = num_chans
                    header_dict['OBSFREQ'] = center_freq * 1e-6
                    header_dict['OBSBW'] = self.sample_rate / self.num_branches * num_chans * 1e-6
                    self._write_header(self._make_header(header_dict), f)
                    
                    v = self.collect_data(num_chans, digitize=digitize, requantize=True)
                    v = v[1:, :][start_chan:start_chan+num_chans, :]
                    
                    if self.num_bits == 8:
                        f.write(np.array(v, dtype=np.int8).tobytes())
                    print(f'Block {j} recorded')
        
