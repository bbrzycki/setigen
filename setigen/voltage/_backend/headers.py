from __future__ import annotations

import pathlib

import numpy as np
from tqdm import tqdm

from .. import raw_utils


def _header_populate_configuration(backend, header_dict=None):
    header_dict = {} if header_dict is None else dict(header_dict)

    if "TELESCOP" not in header_dict:
        header_dict["TELESCOP"] = "SETIGEN"
    elif backend.input_header_dict is not None and "SETIGEN" not in backend.input_header_dict["TELESCOP"]:
        header_dict["TELESCOP"] = f"{backend.input_header_dict['TELESCOP'].strip()}_SETIGEN"
    if "OBSERVER" not in header_dict:
        header_dict["OBSERVER"] = "SETIGEN"
    elif backend.input_header_dict is not None and "SETIGEN" not in backend.input_header_dict["OBSERVER"]:
        header_dict["OBSERVER"] = f"{backend.input_header_dict['OBSERVER'].strip()}_SETIGEN"
    if "SRC_NAME" not in header_dict:
        header_dict["SRC_NAME"] = "SYNTHETIC"
    elif backend.input_header_dict is not None and "SYNTHETIC" not in backend.input_header_dict["SRC_NAME"]:
        header_dict["SRC_NAME"] = f"{backend.input_header_dict['SRC_NAME'].strip()}_SETIGEN"

    header_dict["NBITS"] = backend.num_bits
    header_dict["CHAN_BW"] = backend.chan_bw * 1e-6
    header_dict["NPOL"] = backend.num_pols

    header_dict["BLOCSIZE"] = backend.block_size
    header_dict["SCANLEN"] = backend.obs_length
    header_dict["TBIN"] = backend.tbin
    if backend.is_antenna_array:
        header_dict["NANTS"] = backend.num_antennas
    header_dict["OBSNCHAN"] = backend.num_chans * backend.num_antennas
    header_dict["OBSBW"] = backend.chan_bw * backend.num_chans * 1e-6

    center_freq = (backend.start_chan + (backend.num_chans - 1) / 2) * backend.chan_bw
    center_freq += backend.fch1
    header_dict["OBSFREQ"] = center_freq * 1e-6

    if "PKTIDX" not in header_dict:
        header_dict["PKTIDX"] = 0
    header_dict["PKTIDX"] = int(header_dict["PKTIDX"])
    if "PKTSTART" not in header_dict:
        header_dict["PKTSTART"] = header_dict["PKTIDX"]
    header_dict["PKTSTOP"] = int(header_dict["PKTSTART"]) + backend.num_blocks * backend.samples_per_block

    return header_dict


def _header_add_from_template(header_dict=None):
    header_dict = {} if header_dict is None else dict(header_dict)

    path = pathlib.Path(__file__).resolve().parents[1] / "assets" / "header_template.txt"
    with open(path, "r") as t:
        for line in t.readlines():
            key = line[:8].strip()
            if key != "END" and key not in header_dict:
                header_dict[key] = line[9:].strip()
    return header_dict


def _header_add_from_input_header(input_header_dict, header_dict=None):
    header_dict = {} if header_dict is None else dict(header_dict)

    for key, value in input_header_dict.items():
        if key not in header_dict:
            header_dict[key] = value.strip()
    return header_dict


def _make_header(backend, f, header_dict):
    directio = False

    if "DIRECTIO" in header_dict:
        directio = header_dict["DIRECTIO"]
        try:
            if isinstance(directio, str):
                directio = int(directio.replace("'", ""))
            directio = directio != 0
        except BaseException as err:
            tqdm(f'Could not parse DIRECTIO value `{header_dict["DIRECTIO"]}` ({repr(err)}). Replacing with `0`.')
            header_dict["DIRECTIO"] = 0

    header_lines = 0
    for key, value in header_dict.items():
        value_is_encoded = isinstance(value, str) and value[0] == "'"
        line = raw_utils.format_header_line(key, value, as_strings=value_is_encoded)
        f.write(f"{line:<80}".encode())
        header_lines += 1
    f.write(f"{'END':<80}".encode())
    header_lines += 1

    if directio:
        f.write(bytearray(512 - (80 * header_lines % 512)))

    header_dict["PKTIDX"] += backend.samples_per_block


def _read_next_block(backend):
    _ = backend.input_file_handler.read(backend.header_size)
    data_chunk = backend.input_file_handler.read(backend.block_size)

    obsnchan = backend.num_chans * backend.num_antennas
    rawbuffer = np.frombuffer(data_chunk, dtype=np.int8).reshape((obsnchan, int(backend.block_size / obsnchan)))
    input_voltages = np.zeros((obsnchan, int(rawbuffer.shape[1] / backend.bytes_per_sample * backend.num_pols)),
                              dtype=complex)

    for antenna in range(backend.num_antennas):
        for pol in range(backend.num_pols):
            requantizer = backend.requantizer[antenna][pol]

            c_idx = antenna * backend.num_chans + np.arange(0, backend.num_chans)
            if backend.num_bits == 8:
                t_idx = 2 * pol + np.arange(0, rawbuffer.shape[1], 2 * backend.num_pols)

                r_vals = rawbuffer[c_idx[:, np.newaxis], t_idx[np.newaxis, :]]
                i_vals = rawbuffer[c_idx[:, np.newaxis], (t_idx + 1)[np.newaxis, :]]
            elif backend.num_bits == 4:
                t_idx = pol + np.arange(0, rawbuffer.shape[1], backend.num_pols)

                q_vals = rawbuffer[c_idx[:, np.newaxis], t_idx[np.newaxis, :]]
                r_vals = q_vals // 16
                i_vals = q_vals - 16 * r_vals
                i_vals[i_vals >= 8] -= 16
            else:
                raise ValueError(f"{backend.num_bits} bits not supported...")

            requantizer.quantizer_r._set_target_stats(np.mean(r_vals), np.std(r_vals))
            requantizer.quantizer_i._set_target_stats(np.mean(i_vals), np.std(i_vals))

            t_idx = pol + np.arange(0, input_voltages.shape[1], backend.num_pols)
            input_voltages[c_idx[:, np.newaxis], t_idx[np.newaxis, :]] = r_vals + i_vals * 1j
    return input_voltages
