import numpy as np

from tqdm import tqdm
import time
import copy
from typing import Any
from setigen.voltage import raw_utils
from setigen.voltage import polyphase_filterbank
from setigen.voltage import quantization
from setigen.voltage import antenna as v_antenna
from setigen.voltage._array_backend import xp
from setigen.voltage._backend.headers import (
    _read_next_block,
)
from setigen.voltage._backend.math import (
    _get_block_size,
    _get_total_obs_num_samples,
)
from setigen.voltage._backend.orchestration import (
    _build_record_header,
    _record_files,
    _reset_recording_state,
)
from setigen.voltage._backend.pipeline import (
    _channelize_voltage,
    _get_num_samples_for_subblock,
    _get_subblock_time_range,
    _get_time_indices,
    _get_windows_for_subblock,
    _plan_subblocks,
    _requantize_voltage,
    _split_complex_components,
    _store_complex_components,
)
from setigen.voltage._backend.recording import (
    _RecordConfig,
    _resolve_num_blocks,
)


class RawVoltageBackend(object):
    """
    Central class that wraps around antenna sources and backend elements to facilitate the
    creation of GUPPI RAW voltage files from synthetic real voltages.
    """
    def __init__(self,
                 antenna_source: Any,
                 digitizer: Any,
                 filterbank: Any,
                 requantizer: Any,
                 start_chan: int = 0,
                 num_chans: int = 64,
                 block_size: int = 134217728,
                 blocks_per_file: int = 128,
                 num_subblocks: int = 32,
                 max_working_set_bytes: int = 512 * 1024**2) -> None:
        """Initialize a raw-voltage recording backend.

        Args:
            antenna_source: `Antenna` or `MultiAntennaArray` providing voltage
                samples.
            digitizer: Digitizer template or per-antenna/per-polarization
                digitizer matrix.
            filterbank: PFB template or per-antenna/per-polarization filterbank
                matrix.
            requantizer: Requantizer template or per-antenna/per-polarization
                requantizer matrix.
            start_chan: Index of the first coarse channel to record.
            num_chans: Number of coarse channels to record.
            block_size: Output RAW block size in bytes.
            blocks_per_file: Number of RAW blocks to place in each file.
            num_subblocks: Requested number of computational subblocks per RAW
                block.
            max_working_set_bytes: Soft cap for transient per-subblock working
                set. The backend may increase the effective subblock count to
                stay within this budget.

        Raises:
            ValueError: If the antenna source is invalid.
            TypeError: If a backend element has an unsupported type.
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
        self.max_working_set_bytes = max_working_set_bytes
        
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
                    assert isinstance(self.digitizer[antenna][pol], 
                                      (quantization.RealQuantizer, 
                                       quantization.ComplexQuantizer))
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
                    assert isinstance(self.filterbank[antenna][pol], 
                                      polyphase_filterbank.PolyphaseFilterbank)
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
                    assert isinstance(self.requantizer[antenna][pol], 
                                      quantization.ComplexQuantizer)
        else:
            raise TypeError('Requantizer is incorrect type!')
        self.num_bits = self.requantizer[0][0].num_bits
        self.num_bytes = self.num_bits // 8
        self.bytes_per_sample = 2 * self.num_pols * self.num_bits // 8
        self.total_obs_num_samples = None
        
        # Make sure that block_size is appropriate
        assert self.block_size % int(self.num_antennas * self.num_chans * self.num_taps * self.bytes_per_sample) == 0

        self.samples_per_block = self.block_size // (self.num_antennas * self.num_chans * self.bytes_per_sample)
        self.time_per_block = self.samples_per_block * self.tbin
        
        self.sample_stage_t = 0
        self.digitizer_stage_t = 0
        self.filterbank_stage_t = 0
        self.requantizer_stage_t = 0
        
        self.input_file_stem = None # Filename stem for input RAW data
        self.input_header_dict = None # RAW header
        self.header_size = None # Size of header in file (bytes), including padding
        self.input_num_blocks = None # Total number of blocks in supplied input RAW data
        self.input_file_handler = None # Current file handler for input RAW data
    
    @classmethod
    def from_data(cls, 
                  input_file_stem: str,
                  antenna_source: Any,
                  digitizer: Any = None,
                  filterbank: Any = None,
                  start_chan: int = 0,
                  num_subblocks: int = 32,
                  max_working_set_bytes: int = 512 * 1024**2) -> "RawVoltageBackend":
        """Initialize a backend that injects signals into existing RAW input.

        Args:
            input_file_stem: Path stem to the input RAW data.
            antenna_source: `Antenna` or `MultiAntennaArray` providing injected
                voltage samples.
            digitizer: Optional digitizer template or per-stream matrix.
            filterbank: Optional PFB template or per-stream matrix.
            start_chan: Index of the first coarse channel to record.
            num_subblocks: Requested number of computational subblocks per RAW
                block.
            max_working_set_bytes: Soft cap for transient per-subblock working
                set.

        Returns:
                Initialized backend configured from the input RAW metadata.
        """
        if digitizer is None:
            digitizer = quantization.RealQuantizer()
        if filterbank is None:
            filterbank = polyphase_filterbank.PolyphaseFilterbank()
        requantizer = quantization.ComplexQuantizer()
        
        raw_params = raw_utils.get_raw_params(input_file_stem=input_file_stem,
                                              start_chan=start_chan)
        
        blocks_per_file = raw_utils.get_blocks_per_file(input_file_stem)
        
        requantizer.num_bits = raw_params['num_bits']
        requantizer.quantizer_r.num_bits = raw_params['num_bits']
        requantizer.quantizer_i.num_bits = raw_params['num_bits']
        # if isinstance(requantizer, quantization.ComplexQuantizer):
        #     requantizer.num_bits = raw_params['num_bits']
        #     requantizer.quantizer_r.num_bits = raw_params['num_bits']
        #     requantizer.quantizer_i.num_bits = raw_params['num_bits']
        # elif isinstance(requantizer, list):
        #     for r in requantizer:
        #         r.num_bits = raw_params['num_bits']
        #         r.quantizer_r.num_bits = raw_params['num_bits']
        #         r.quantizer_i.num_bits = raw_params['num_bits']
        
        backend = cls(antenna_source,
                      digitizer=digitizer,
                      filterbank=filterbank,
                      requantizer=requantizer,
                      start_chan=start_chan,
                      num_chans=raw_params['num_chans'],
                      block_size=raw_params['block_size'],
                      blocks_per_file=blocks_per_file,
                      num_subblocks=num_subblocks,
                      max_working_set_bytes=max_working_set_bytes)
        
        backend.input_file_stem = input_file_stem   
        backend.input_num_blocks = raw_utils.get_total_blocks(input_file_stem)
        backend.input_header_dict = raw_utils.read_header(f'{input_file_stem}.0000.raw')

        if int(backend.input_header_dict.get("DIRECTIO", 0)) == 0:
            backend.header_size = 80 * (len(backend.input_header_dict) + 1)
        else:
            backend.header_size = int(512 * np.ceil((80 * (len(backend.input_header_dict) + 1)) / 512))
        
        return backend
    
    def collect_data_block(self,
                           digitize: bool = True,
                           requantize: bool = True,
                           verbose: bool = True) -> np.ndarray:
        """Collect one packed RAW data block from the antenna source.

        Args:
            digitize: Whether to quantize input voltages before the PFB.
            requantize: Whether to quantize complex post-PFB voltages.
            verbose: Whether to emit progress output.

        Returns:
                Packed RAW buffer with shape `(num_chans * num_antennas,
                block_size / (num_chans * num_antennas))`.

        Raises:
            ValueError: If input RAW mixing is requested without requantization.
        """
        obsnchan = self.num_chans * self.num_antennas
        if requantize:
            output_dtype = np.int8 if self.num_bits == 8 else np.uint8
        else:
            output_dtype = np.float32
        final_voltages = np.empty((obsnchan, int(self.block_size / obsnchan)),
                                  dtype=output_dtype)
        
        if self.input_file_stem is not None:
            if not requantize:
                raise ValueError("Must set 'requantize=True' when using input RAW data!")
            input_voltages = _read_next_block(self)
        else:
            input_voltages = None

        plan = _plan_subblocks(self, obsnchan=obsnchan)
        self.num_subblocks = plan.num_subblocks
        
        with tqdm(total=self.num_antennas*self.num_pols*self.num_subblocks,
                  leave=False,
                  disable=not verbose) as pbar:
            pbar.set_description('Subblocks')

            for subblock in range(self.num_subblocks):
                if verbose:
                    tqdm.write(f'Creating subblock {subblock}...')

                windows = _get_windows_for_subblock(plan, self, subblock)
                num_samples = _get_num_samples_for_subblock(self, windows=windows)
                    
                # Calculate the real voltage samples from each antenna
                t = time.time()
                antennas_v = self.antenna_source.get_samples(num_samples)
                self.sample_stage_t += time.time() - t

                for antenna in range(self.num_antennas):
                    if verbose and self.is_antenna_array:
                        tqdm.write(f'Creating antenna #{antenna}...')
                        
                    c_idx = antenna * self.num_chans + np.arange(0, self.num_chans)
                    for pol in range(self.num_pols):
                        subblock_t_range = _get_subblock_time_range(plan,
                                                                    self,
                                                                    subblock=subblock,
                                                                    windows=windows)
                        t_idx = _get_time_indices(self,
                                                  subblock=subblock,
                                                  bytes_per_subblock=plan.bytes_per_subblock,
                                                  subblock_time_range=subblock_t_range,
                                                  pol=pol)
                        
                        # Send voltage data through the backend
                        v = antennas_v[antenna][pol]
                        v = _channelize_voltage(self,
                                                v,
                                                antenna=antenna,
                                                pol=pol,
                                                digitize=digitize)

                        if requantize:
                            v = _requantize_voltage(self,
                                                    v,
                                                    antenna=antenna,
                                                    pol=pol,
                                                    c_idx=c_idx,
                                                    t_idx=t_idx,
                                                    input_voltages=input_voltages,
                                                    digitize=digitize,
                                                    xp=xp)

                        R, I = _split_complex_components(v, xp=xp)
                        _store_complex_components(self,
                                                  final_voltages,
                                                  c_idx=c_idx,
                                                  t_idx=t_idx,
                                                  real=R,
                                                  imag=I,
                                                  requantize=requantize)

                        pbar.update(1)
            
        return final_voltages    
    
    
    def get_num_blocks(self, obs_length: float) -> int:
        """Calculate the number of RAW blocks implied by an observation length.

        Args:
            obs_length: Observation length in seconds.

        Returns:
                Number of full RAW blocks required.
        """
        return int(obs_length * abs(self.chan_bw) * self.num_antennas * self.num_chans * self.bytes_per_sample / self.block_size)
        
    
    def record(self, 
               output_file_stem: str,
               obs_length: float | None = None, 
               num_blocks: int | None = None,
               length_mode: str = 'obs_length',
               header_dict: dict[str, Any] | None = None,
               digitize: bool = True,
               load_template: bool = True,
               verbose: bool = True) -> None:
        """Record one observation as one or more GUPPI RAW files.

        Args:
            output_file_stem: Path stem for the output RAW files.
            obs_length: Observation length in seconds when using
                `length_mode="obs_length"`.
            num_blocks: Number of RAW blocks when using
                `length_mode="num_blocks"`.
            length_mode: Strategy for interpreting the supplied observation
                length.
            header_dict: Optional RAW header overrides and additions.
            digitize: Whether to quantize input voltages before the PFB.
            load_template: Whether to merge the built-in RAW header template.
            verbose: Whether to emit progress output.
        """
        record_config = _RecordConfig.from_values(obs_length=obs_length,
                                                  num_blocks=num_blocks,
                                                  length_mode=length_mode,
                                                  header_dict=header_dict,
                                                  digitize=digitize,
                                                  load_template=load_template,
                                                  verbose=verbose)
        self.num_blocks = _resolve_num_blocks(record_config.length,
                                              get_num_blocks=self.get_num_blocks,
                                              fallback_num_blocks=self.input_num_blocks)
        
        # Ensure that we don't request more blocks than possible
        if self.input_num_blocks is not None:
            self.num_blocks = min(self.num_blocks, self.input_num_blocks)
            
        self.obs_length = self.num_blocks * self.time_per_block
        self.total_obs_num_samples = int(self.obs_length / self.tbin) * self.num_branches
        
        header_dict = _build_record_header(self, record_config)
        _reset_recording_state(self)
        _record_files(self,
                      output_file_stem=output_file_stem,
                      record_config=record_config,
                      header_dict=header_dict,
                      xp=xp)

    def to_spectrogram(
        self,
        spec: Any,
        obs_length: float | None = None,
        num_blocks: int | None = None,
        length_mode: str = "obs_length",
        digitize: bool = True,
        requantize: bool = True,
        verbose: bool = True,
    ) -> Any:
        """Generate a reduced spectrogram directly from this backend.

        This follows the same voltage path as :meth:`record` through sample
        generation, digitization, coarse PFB channelization, and requantization,
        then reduces directly to a spectrogram without packing, writing,
        reading, or decoding an intermediate RAW file.

        Args:
            spec: :class:`setigen.voltage.VoltageSpectrogramSpec` instance.
            obs_length: Observation length in seconds when using
                ``length_mode="obs_length"``.
            num_blocks: Number of backend blocks when using
                ``length_mode="num_blocks"``.
            length_mode: Strategy for interpreting the supplied observation
                length.
            digitize: Whether to quantize input voltages before the PFB.
            requantize: Whether to quantize complex post-PFB voltages.
            verbose: Whether to emit progress output.

        Returns:
            :class:`setigen.voltage.VoltageSpectrogramResult`.
        """
        from setigen.voltage.spectrogram import generate_voltage_spectrogram

        return generate_voltage_spectrogram(
            self,
            spec,
            obs_length=obs_length,
            num_blocks=num_blocks,
            length_mode=length_mode,
            digitize=digitize,
            requantize=requantize,
            verbose=verbose,
            xp=xp,
        )
                    
             
def get_block_size(num_antennas: int = 1,
                   tchans_per_block: int = 128,
                   num_bits: int = 8,
                   num_pols: int = 2,
                   num_branches: int = 1024,
                   num_chans: int = 64,
                   fftlength: int = 1024,
                   int_factor: int = 4) -> int:
    """Calculate a RAW block size from the desired reduced-product dimensions.

    Args:
        num_antennas: Number of antennas.
        tchans_per_block: Number of output time bins per RAW block after fine
            channelization.
        num_bits: Requantized bit depth, usually 8 or 4.
        num_pols: Number of recorded polarizations.
        num_branches: Number of PFB branches.
        num_chans: Number of recorded coarse channels.
        fftlength: Fine-channel FFT length.
        int_factor: Fine-channel time integration factor.

    Returns:
            RAW block size in bytes.
    """
    return _get_block_size(num_antennas=num_antennas,
                           tchans_per_block=tchans_per_block,
                           num_bits=num_bits,
                           num_pols=num_pols,
                           num_branches=num_branches,
                           num_chans=num_chans,
                           fftlength=fftlength,
                           int_factor=int_factor)


def get_total_obs_num_samples(obs_length: float | None = None, 
                              num_blocks: int | None = None, 
                              length_mode: str = 'obs_length',
                              num_antennas: int = 1,
                              sample_rate: float = 3e9,
                              block_size: int = 134217728,
                              num_bits: int = 8,
                              num_pols: int = 2,
                              num_branches: int = 1024,
                              num_chans: int = 64) -> int:
    """Estimate the required real-voltage sample count without constructing a backend.

    Args:
        obs_length: Observation length in seconds when using
            `length_mode="obs_length"`.
        num_blocks: Number of RAW blocks when using
            `length_mode="num_blocks"`.
        length_mode: Strategy for interpreting the supplied observation length.
        num_antennas: Number of antennas.
        sample_rate: Time-domain sample rate in Hz.
        block_size: RAW block size in bytes.
        num_bits: Requantized bit depth.
        num_pols: Number of recorded polarizations.
        num_branches: Number of PFB branches.
        num_chans: Number of recorded coarse channels.

    Returns:
            Estimated count of real voltage samples required.
    """
    return _get_total_obs_num_samples(obs_length=obs_length,
                                      num_blocks=num_blocks,
                                      length_mode=length_mode,
                                      num_antennas=num_antennas,
                                      sample_rate=sample_rate,
                                      block_size=block_size,
                                      num_bits=num_bits,
                                      num_pols=num_pols,
                                      num_branches=num_branches,
                                      num_chans=num_chans)
