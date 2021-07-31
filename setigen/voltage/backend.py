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
import copy
import glob

from setigen import unit_utils
from . import raw_utils
from . import polyphase_filterbank
from . import quantization
from . import antenna as v_antenna


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
        digitizer : RealQuantizer or ComplexQuantizer, or list, optional
            Quantizer used to digitize input voltages. Either a single object to be used as a template
            for each antenna and polarization, or a 2D list of quantizers of shape (num_antennas, num_pols).
        filterbank : PolyphaseFilterbank, or list, optional
            Polyphase filterbank object used to channelize voltages. Either a single object to be used as a 
            template for each antenna and polarization, or a 2D list of filterbank objects of shape 
            (num_antennas, num_pols).
        requantizer : ComplexQuantizer, or list, optional
            Quantizer used on complex channelized voltages. Either a single object to be used as a template
            for each antenna and polarization, or a 2D list of quantizers of shape (num_antennas, num_pols).
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
        if isinstance(antenna_source, v_antenna.Antenna):
            self.num_antennas = 1
            self.is_antenna_array = False
        elif isinstance(antenna_source, v_antenna.MultiAntennaArray):
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
        if isinstance(self.digitizer, quantization.RealQuantizer) or isinstance(self.digitizer, quantization.ComplexQuantizer):
            self.digitizer = [[copy.deepcopy(self.digitizer) 
                               for pol in range(self.num_pols)]
                              for antenna in range(self.num_antennas)]
        elif isinstance(self.digitizer, list):
            assert len(self.digitizer) == self.num_antennas
            assert len(self.digitizer[0]) == self.num_pols
            for antenna in range(self.num_antennas):
                for pol in range(self.num_pols):
                    digitizer = self.digitizer[antenna][pol]
                    assert isinstance(digitizer, quantization.RealQuantizer) or isinstance(digitizer, quantization.ComplexQuantizer)
        else:
            raise TypeError('Digitizer is incorrect type!')
        
        self.filterbank = filterbank
        if isinstance(self.filterbank, polyphase_filterbank.PolyphaseFilterbank):
            self.filterbank = [[copy.deepcopy(self.filterbank) 
                                for pol in range(self.num_pols)]
                               for antenna in range(self.num_antennas)]
        elif isinstance(self.filterbank, list):
            assert len(self.filterbank) == self.num_antennas
            assert len(self.filterbank[0]) == self.num_pols
            for antenna in range(self.num_antennas):
                for pol in range(self.num_pols):
                    filterbank = self.filterbank[antenna][pol]
                    assert isinstance(self.filterbank, polyphase_filterbank.PolyphaseFilterbank)
        else:
            raise TypeError('Filterbank is incorrect type!')
            
        self.num_taps = self.filterbank[0][0].num_taps
        self.num_branches = self.filterbank[0][0].num_branches
        assert self.start_chan + self.num_chans <= self.num_branches // 2
        
        self.tbin = self.num_branches / self.sample_rate
        self.chan_bw = 1 / self.tbin
        if not self.ascending:
            self.chan_bw = -self.chan_bw
        
        self.requantizer = requantizer
        if isinstance(self.requantizer, quantization.ComplexQuantizer):
            self.requantizer = [[copy.deepcopy(self.requantizer) 
                                 for pol in range(self.num_pols)]
                                for antenna in range(self.num_antennas)]
        elif isinstance(self.requantizer, list):
            assert len(self.requantizer) == self.num_antennas
            assert len(self.requantizer[0]) == self.num_pols
            for antenna in range(self.num_antennas):
                for pol in range(self.num_pols):
                    requantizer = self.requantizer[antenna][pol]
                    assert isinstance(self.requantizer, quantization.ComplexQuantizer)
        else:
            raise TypeError('Requantizer is incorrect type!')
        self.num_bits = self.requantizer[0][0].num_bits
        self.num_bytes = self.num_bits // 8
        self.bytes_per_sample = 2 * self.num_pols * self.num_bits // 8
        self.total_obs_num_samples = None
        
        self.time_per_block = self.block_size / (self.num_antennas * self.num_chans * self.bytes_per_sample) * self.tbin
    
        # Make sure that block_size is appropriate
        assert self.block_size % int(self.num_antennas * self.num_chans * self.num_taps * self.bytes_per_sample) == 0
        
        self.sample_stage_t = 0
        self.digitizer_stage_t = 0
        self.filterbank_stage_t = 0
        self.requantizer_stage_t = 0
        
        self.input_file_stem = None # Filename stem for input RAW data
        self.input_header_dict = None # RAW header
        self.header_size = None # Size of header in file (bytes)
        self.input_num_blocks = None # Total number of blocks in supplied input RAW data
        self.input_file_handler = None # Current file handler for input RAW data
        
    @classmethod
    def from_data(cls, 
                  input_file_stem,
                  antenna_source,
                  digitizer=quantization.RealQuantizer(),
                  filterbank=polyphase_filterbank.PolyphaseFilterbank(),
#                   requantizer=quantization.ComplexQuantizer(),
                  start_chan=0,
                  num_subblocks=32):
        """
        Initialize a RawVoltageBackend object, using existing RAW data as a background for
        signal insertion and recording. Compared to normal initialization, some parameters are inferred 
        from the input data.

        Parameters
        ----------
        input_file_stem : str
            Filename or path stem to input RAW data
        antenna_source : Antenna or MultiAntennaArray
            Antenna or MultiAntennaArray, from which real voltage data is created
        digitizer : RealQuantizer or ComplexQuantizer, or list, optional
            Quantizer used to digitize input voltages. Either a single object to be used as a template
            for each antenna and polarization, or a 2D list of quantizers of shape (num_antennas, num_pols).
        filterbank : PolyphaseFilterbank, or list, optional
            Polyphase filterbank object used to channelize voltages. Either a single object to be used as a 
            template for each antenna and polarization, or a 2D list of filterbank objects of shape 
            (num_antennas, num_pols).
        start_chan : int, optional
            Index of first coarse channel to be recorded
        num_subblocks : int, optional
            Number of partitions per block, used for computation. If `num_subblocks`=1, one block's worth
            of data will be passed through the pipeline and recorded at once. Use this parameter to reduce 
            memory load, especially when using GPU acceleration.
            
        Returns
        -------
        backend : RawVoltageBackend
            Created backend object
        """
        requantizer=quantization.ComplexQuantizer()
        
        raw_params = raw_utils.get_raw_params(input_file_stem=input_file_stem,
                                              start_chan=start_chan)
        
        blocks_per_file = raw_utils.get_blocks_per_file(input_file_stem)
        
        if isinstance(requantizer, quantization.ComplexQuantizer):
            requantizer.num_bits = raw_params['num_bits']
            requantizer.quantizer_r.num_bits = raw_params['num_bits']
            requantizer.quantizer_i.num_bits = raw_params['num_bits']
        elif isinstance(requantizer, list):
            for r in requantizer:
                r.num_bits = raw_params['num_bits']
                r.quantizer_r.num_bits = raw_params['num_bits']
                r.quantizer_i.num_bits = raw_params['num_bits']
        
        backend = cls(antenna_source,
                      digitizer=digitizer,
                      filterbank=filterbank,
                      requantizer=requantizer,
                      start_chan=start_chan,
                      num_chans=raw_params['num_chans'],
                      block_size=raw_params['block_size'],
                      blocks_per_file=blocks_per_file,
                      num_subblocks=num_subblocks)
        
        backend.input_file_stem = input_file_stem   
        backend.input_num_blocks = raw_utils.get_total_blocks(input_file_stem)
        backend.input_header_dict = raw_utils.read_header(f'{input_file_stem}.0000.raw')
        backend.header_size = int(512 * np.ceil((80 * (len(backend.input_header_dict) + 1)) / 512))
        
        return backend
    
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
            # Exclude END line
            template_lines = t.readlines()[:-1]
            
        # Set header values determined by pipeline parameters
        if 'TELESCOP' not in header_dict:
            header_dict['TELESCOP'] = 'SETIGEN'
            if self.input_header_dict is not None:
                header_dict['TELESCOP'] = f"{self.input_header_dict['TELESCOP']}_SETIGEN"
        if 'OBSERVER' not in header_dict:
            header_dict['OBSERVER'] = 'SETIGEN'
            if self.input_header_dict is not None:
                header_dict['OBSERVER'] = f"{self.input_header_dict['OBSERVER']}_SETIGEN"
        if 'SRC_NAME' not in header_dict:
            header_dict['SRC_NAME'] = 'SYNTHETIC'
            if self.input_header_dict is not None:
                header_dict['SRC_NAME'] = f"{self.input_header_dict['SRC_NAME']}_SETIGEN"
        
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
        header_dict['OBSNCHAN'] = self.num_chans * self.num_antennas
        header_dict['OBSBW'] = self.chan_bw * self.num_chans * 1e-6

        # Compute center frequency of recorded data
        # self.chan_bw was already adjusted for ascending or descending frequencies 
        center_freq = (self.start_chan + (self.num_chans - 1) / 2) * self.chan_bw
        center_freq += self.fch1
        header_dict['OBSFREQ'] = center_freq * 1e-6
        
        header_lines = []
        used_keys = set()
        for i in range(len(template_lines)):
            key = template_lines[i][:8].strip()
            used_keys.add(key)
            if key in header_dict:
                header_lines.append(raw_utils.format_header_line(key, 
                                                                 header_dict[key],
                                                                 as_strings=False))
            elif self.input_header_dict is not None:
                # If initialized with raw files, use their header values
                header_lines.append(raw_utils.format_header_line(key, 
                                                                 self.input_header_dict[key], 
                                                                 as_strings=True))
            else:
                # Otherwise use template header key, value pairs; cut newline character
                header_lines.append(template_lines[i][:-1])
        # Add new key value pairs
        for key in header_dict.keys() - used_keys:
            header_lines.append(raw_utils.format_header_line(key, header_dict[key]))
        header_lines.append(f"{'END':<80}")
        
        # Write each line with space and zero padding
        for line in header_lines:
            f.write(f"{line:<80}".encode())
        f.write(bytearray(512 - (80 * len(header_lines) % 512)))   
        
    def _read_next_block(self):
        """
        Reads next block of data if input RAW files are provided, upon which synthetic data will 
        be added. Also sets requantizer target statistics appropriately.
        """
        _ = self.input_file_handler.read(self.header_size)
        data_chunk = self.input_file_handler.read(self.block_size)
        
        obsnchan = self.num_chans * self.num_antennas
        rawbuffer = np.frombuffer(data_chunk, dtype=np.int8).reshape((obsnchan, int(self.block_size / obsnchan)))
        input_voltages = np.zeros((obsnchan, int(rawbuffer.shape[1] / self.bytes_per_sample * self.num_pols)), 
                                  dtype=complex)
        
        for antenna in range(self.num_antennas):
            for pol in range(self.num_pols):
                requantizer = self.requantizer[antenna][pol]
                
                c_idx = antenna * self.num_chans + np.arange(0, self.num_chans)
                if self.num_bits == 8:
                    t_idx = 2 * pol + np.arange(0, rawbuffer.shape[1], 2 * self.num_pols)
                    
                    R = rawbuffer[c_idx[:, np.newaxis], t_idx[np.newaxis, :]]
                    I = rawbuffer[c_idx[:, np.newaxis], (t_idx+1)[np.newaxis, :]]
                elif self.num_bits == 4:
                    t_idx = pol + np.arange(0, rawbuffer.shape[1], self.num_pols)
                    
                    Q = rawbuffer[c_idx[:, np.newaxis], t_idx[np.newaxis, :]]
                    R = Q // 16
                    I = Q - 16 * R
                    I[I >= 8] -= 16
                else:
                    raise ValueError(f'{self.num_bits} bits not supported...')
                
                requantizer.quantizer_r._set_target_stats(np.mean(R), np.std(R))
                requantizer.quantizer_i._set_target_stats(np.mean(I), np.std(I))
                
                t_idx = pol + np.arange(0, input_voltages.shape[1], self.num_pols)
                input_voltages[c_idx[:, np.newaxis], t_idx[np.newaxis, :]] = R + I * 1j
        return input_voltages
        
    def collect_data_block(self,
                           digitize=True,
                           requantize=True,
                           verbose=True):
        """
        General function to actually collect data from the antenna source and return coarsely channelized 
        complex voltages. Collects one block of data.
        
        Parameters
        ----------
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
        obsnchan = self.num_chans * self.num_antennas
        final_voltages = np.empty((obsnchan, int(self.block_size / obsnchan)))
        
        if self.input_file_stem is not None:
            if not requantize:
                raise ValueError("Must set 'requantize=True' when using input RAW data!")
            input_voltages = self._read_next_block()
    
        # Make sure that block_size is appropriate
        assert self.block_size % int(obsnchan * self.num_taps * self.bytes_per_sample) == 0
        T = int(self.block_size / (obsnchan * self.bytes_per_sample))

        W = int(xp.ceil(T / self.num_taps / self.num_subblocks)) + 1
        subblock_T = self.num_taps * (W - 1)

        # Change self.num_subblocks if necessary
        self.num_subblocks = int(xp.ceil(T / subblock_T))
        subblock_t_len = int(subblock_T * self.bytes_per_sample)
        
        with tqdm(total=self.num_antennas*self.num_pols*self.num_subblocks, leave=False) as pbar:
            pbar.set_description('Subblocks')

            for subblock in range(self.num_subblocks):
                if verbose:
                    tqdm.write(f'Creating subblock {subblock}...')

                # Change num windows at the end if self.num_subblocks doesn't go in evenly
                if T % subblock_T != 0 and subblock == self.num_subblocks - 1:
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
                        
                    c_idx = antenna * self.num_chans + np.arange(0, self.num_chans)
                    for pol in range(self.num_pols):
                        # Store indices used for numpy data i/o per polarization
                        if T % subblock_T != 0 and subblock == self.num_subblocks - 1:
                            # Uses smaller num windows W
                            subblock_t_range = self.num_taps * (W - 1) * self.bytes_per_sample
                        else:
                            subblock_t_range = subblock_t_len
                        t_idx = subblock * subblock_t_len + self.num_bits // 4 * pol + np.arange(0,
                                                                                                 subblock_t_range,
                                                                                                 self.num_bits // 4 * self.num_pols)
                        
                        # Send voltage data through the backend
                        v = antennas_v[antenna][pol]

                        if digitize:
                            t = time.time()
                            v = self.digitizer[antenna][pol].quantize(v)
                            self.digitizer_stage_t += time.time() - t

                        t = time.time()
                        v = self.filterbank[antenna][pol].channelize(v)
                        v = v[:, self.start_chan:self.start_chan+self.num_chans]
                        self.filterbank_stage_t += time.time() - t

                        if requantize:
                            t = time.time()
                            
                            if self.input_file_stem is not None:
                                temp_mean_r = self.requantizer[antenna][pol].quantizer_r.target_mean
                                self.requantizer[antenna][pol].quantizer_r.target_mean = 0
                                temp_mean_i = self.requantizer[antenna][pol].quantizer_i.target_mean
                                self.requantizer[antenna][pol].quantizer_i.target_mean = 0
    
                                # Start off assuming signals are embedded in Gaussian noise with std 1
                                custom_stds = self.filterbank[antenna][pol].channelized_stds
                                # If digitizing real voltages, scale up by the appropriate factor
                                if digitize:
                                    custom_stds *= self.digitizer[antenna][pol].target_std
                                v = self.requantizer[antenna][pol].quantize(v, custom_stds=custom_stds)
                                
                                self.requantizer[antenna][pol].quantizer_r.target_mean = temp_mean_r
                                self.requantizer[antenna][pol].quantizer_i.target_mean = temp_mean_i
                                
                                if self.num_bits == 8:
                                    input_v = input_voltages[c_idx[:, np.newaxis], (t_idx//2)[np.newaxis, :]]
                                elif self.num_bits == 4:
                                    input_v = input_voltages[c_idx[:, np.newaxis], t_idx[np.newaxis, :]]
                                input_v = xp.array(input_v)
                                v += input_v.T
                                
                            v = self.requantizer[antenna][pol].quantize(v)
                            self.requantizer_stage_t += time.time() - t

                        # Convert to numpy array if using cupy
                        try:
                            R = xp.asnumpy(xp.real(v).T)  
                            I = xp.asnumpy(xp.imag(v).T)  
                        except AttributeError:
                            R = xp.real(v).T
                            I = xp.imag(v).T
                            
                        if self.num_bits == 8 or not requantize:
                            final_voltages[c_idx[:, np.newaxis], t_idx[np.newaxis, :]] = R
                            final_voltages[c_idx[:, np.newaxis], (t_idx+1)[np.newaxis, :]] = I
                        elif self.num_bits == 4:
                            # Translate 4 bit complex voltages to an 8 bit equivalent representation
                            I[I < 0] += 16
                            final_voltages[c_idx[:, np.newaxis], t_idx[np.newaxis, :]] = R * 16 + I
                        else:
                            raise ValueError(f'{self.num_bits} bits not supported...')

                        pbar.update(1)
            
        return final_voltages    
    
    def get_num_blocks(self, obs_length):
        """
        Calculate the number of blocks required as a function of observation length, in seconds. Note that only 
        an integer number of blocks will be recorded, so the actual observation length may be shorter than the 
        `obs_length` provided.
        """
        return int(obs_length * abs(self.chan_bw) * self.num_antennas * self.num_chans * self.bytes_per_sample / self.block_size)
        
    def record(self, 
               output_file_stem,
               obs_length=None, 
               num_blocks=None,
               length_mode='obs_length',
               header_dict={},
               digitize=True,
               verbose=True):
        """
        General function to actually collect data from the antenna source and return coarsely channelized complex
        voltages. If input data is provided, only as much data as is in the input will be generated.
        
        Parameters
        ----------
        output_file_stem : str
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
                if self.input_num_blocks is not None:
                    self.num_blocks = self.input_num_blocks
                else:
                    raise ValueError("Value not given for 'obs_length'.")
            else:
                self.num_blocks = self.get_num_blocks(obs_length)
        elif length_mode == 'num_blocks':
            if num_blocks is None:
                if self.input_num_blocks is not None:
                    self.num_blocks = self.input_num_blocks
                else:
                    raise ValueError("Value not given for 'num_blocks'.")
            else:
                self.num_blocks = num_blocks
        else:
            raise ValueError("Invalid option given for 'length_mode'.")
            
        # Ensure that we don't request more blocks than possible
        if self.input_num_blocks is not None:
            self.num_blocks = min(self.num_blocks, self.input_num_blocks)
            
        self.obs_length = self.num_blocks * self.time_per_block
        self.total_obs_num_samples = int(self.obs_length / self.tbin) * self.num_branches
        
        # Mark each antenna and data stream as the start of the observation
        self.antenna_source.reset_start()
        
        # Reset filterbank cache as well
        for antenna in range(self.num_antennas):
            for pol in range(self.num_pols):
                self.digitizer[antenna][pol]._reset_cache()
                self.filterbank[antenna][pol]._reset_cache()
                self.requantizer[antenna][pol]._reset_cache()
            
        # Collect data and record to disk
        num_files = int(xp.ceil(self.num_blocks / self.blocks_per_file))
        with tqdm(total=self.num_blocks) as pbar:
            pbar.set_description('Blocks')
            for i in range(num_files):
                save_fn = f'{output_file_stem}.{i:04}.raw'
                # Create input raw file handler for use in collecting data
                if self.input_file_stem is not None:
                    input_fn = f'{self.input_file_stem}.{i:04}.raw'
                    self.input_file_handler = open(input_fn, 'rb')
                with open(save_fn, 'wb') as f:
                    # If blocks won't fill a whole file, adjust number of blocks to write at the end
                    if i == num_files - 1 and self.num_blocks % self.blocks_per_file != 0:
                        blocks_to_write = self.num_blocks % self.blocks_per_file
                    else:
                        blocks_to_write = self.blocks_per_file
                    for j in range(blocks_to_write):
                        if verbose:
                            tqdm.write(f'Creating block {j}...')
                        self._make_header(f, header_dict)
                        v = self.collect_data_block(digitize=digitize, 
                                                    requantize=True,
                                                    verbose=verbose)

                        f.write(xp.array(v, dtype=xp.int8).tobytes())
                        if verbose:
                            tqdm.write(f'File {i}, block {j} recorded!')
                        pbar.update(1)
                if self.input_file_stem is not None:
                    self.input_file_handler.close()
                    
                    
def get_block_size(num_antennas=1,
                   tchans_per_block=128,
                   num_bits=8,
                   num_pols=2,
                   num_branches=1024,
                   num_chans=64,
                   fftlength=1024,
                   int_factor=4):
    """
    Calculate block size, given a desired number of time bins per RAW data block 
    `tchans_per_block`. Takes in backend parameters, including fine channelization
    factors. Can be used to calculate reasonable block sizes for raw voltage recording.
    
    Parameters
    ----------
    num_antennas : int
        Number of antennas
    tchans_per_block : int
        Final number of time bins in fine resolution product, per data block
    num_bits : int
        Number of bits in requantized data (for saving into file). Can be 8 or 4.
    num_pols : int
        Number of polarizations recorded
    num_branches : int
        Number of branches in polyphase filterbank 
    num_chans : int
        Number of coarse channels written to file
    fftlength : int
        FFT length to be used in fine channelization
    int_factor : int, optional
        Integration factor to be used in fine channelization
    
    Returns
    -------
    block_size : int
        Block size, in bytes
    """
    obsnchan = num_chans * num_antennas
    bytes_per_sample = 2 * num_pols * num_bits // 8
    T = tchans_per_block * fftlength * int_factor
    block_size = T * obsnchan * bytes_per_sample
    return block_size

                    
def get_total_obs_num_samples(obs_length=None, 
                              num_blocks=None, 
                              length_mode='obs_length',
                              num_antennas=1,
                              sample_rate=3e9,
                              block_size=134217728,
                              num_bits=8,
                              num_pols=2,
                              num_branches=1024,
                              num_chans=64):
    """
    Calculate number of required real voltage time samples for as given `obs_length` or `num_blocks`, without directly 
    using a `RawVoltageBackend` object. 
    
    Parameters
    ----------
    obs_length : float, optional
        Length of observation in seconds, if in `obs_length` mode
    num_blocks : int, optional
        Number of data blocks to record, if in `num_blocks` mode
    length_mode : str, optional
        Mode for specifying length of observation, either `obs_length` in seconds or `num_blocks` in data blocks
    num_antennas : int
        Number of antennas
    sample_rate : float
        Sample rate in Hz
    block_size : int
        Block size used in recording GUPPI RAW files
    num_bits : int
        Number of bits in requantized data (for saving into file). Can be 8 or 4.
    num_pols : int
        Number of polarizations recorded
    num_branches : int
        Number of branches in polyphase filterbank 
    num_chans : int
        Number of coarse channels written to file
    
    Returns
    -------
    num_samples : int
        Number of samples
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
    return num_blocks * int(block_size / (num_antennas * num_chans * bytes_per_sample)) * num_branches


