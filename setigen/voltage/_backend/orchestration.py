from __future__ import annotations

from contextlib import nullcontext

from tqdm import tqdm

from .headers import (
    _header_add_from_input_header,
    _header_add_from_template,
    _header_populate_configuration,
    _make_header,
)


def _build_record_header(backend, record_config):
    header_dict = dict(record_config.header_dict)

    if record_config.load_template:
        header_dict = _header_add_from_template(header_dict)
    if backend.input_header_dict is not None:
        header_dict = _header_add_from_input_header(backend.input_header_dict, header_dict)
    return _header_populate_configuration(backend, header_dict)


def _reset_recording_state(backend):
    backend.antenna_source.reset_start()

    for antenna in range(backend.num_antennas):
        for pol in range(backend.num_pols):
            backend.digitizer[antenna][pol]._reset_cache()
            backend.filterbank[antenna][pol]._reset_cache()
            backend.requantizer[antenna][pol]._reset_cache()


def _get_num_output_files(backend, *, xp):
    return int(xp.ceil(backend.num_blocks / backend.blocks_per_file))


def _get_blocks_to_write(backend, *, file_index, num_files):
    if file_index == num_files - 1 and backend.num_blocks % backend.blocks_per_file != 0:
        return backend.num_blocks % backend.blocks_per_file
    return backend.blocks_per_file


def _record_files(backend,
                  *,
                  output_file_stem,
                  record_config,
                  header_dict,
                  xp):
    num_files = _get_num_output_files(backend, xp=xp)
    with tqdm(total=backend.num_blocks) as pbar:
        pbar.set_description("Blocks")
        for file_index in range(num_files):
            save_fn = f"{output_file_stem}.{file_index:04}.raw"

            input_context = nullcontext(None)
            if backend.input_file_stem is not None:
                input_fn = f"{backend.input_file_stem}.{file_index:04}.raw"
                input_context = open(input_fn, "rb")

            with input_context as input_file_handler:
                backend.input_file_handler = input_file_handler
                with open(save_fn, "wb") as f:
                    blocks_to_write = _get_blocks_to_write(backend,
                                                           file_index=file_index,
                                                           num_files=num_files)
                    for block_index in range(blocks_to_write):
                        if record_config.verbose:
                            tqdm.write(f"Creating block {block_index}...")
                        _make_header(backend, f, header_dict)
                        voltages = backend.collect_data_block(digitize=record_config.digitize,
                                                              requantize=True,
                                                              verbose=record_config.verbose)
                        f.write(xp.array(voltages, dtype=xp.int8).tobytes())
                        if record_config.verbose:
                            tqdm.write(f"File {file_index}, block {block_index} recorded!")
                        pbar.update(1)
                backend.input_file_handler = None
