from __future__ import annotations

from typing import Any, BinaryIO

import numpy as np
from tqdm import tqdm

from .. import raw_utils


_HEADER_KEY_BACKEND = "BACKEND"
_HEADER_KEY_TELESCOP = "TELESCOP"
_HEADER_KEY_OBSERVER = "OBSERVER"
_HEADER_KEY_SRC_NAME = "SRC_NAME"
_HEADER_KEY_OBS_MODE = "OBS_MODE"
_HEADER_KEY_PKTFMT = "PKTFMT"
_HEADER_KEY_NBITS = "NBITS"
_HEADER_KEY_CHAN_BW = "CHAN_BW"
_HEADER_KEY_NPOL = "NPOL"
_HEADER_KEY_BLOCSIZE = "BLOCSIZE"
_HEADER_KEY_SCANLEN = "SCANLEN"
_HEADER_KEY_TBIN = "TBIN"
_HEADER_KEY_NANTS = "NANTS"
_HEADER_KEY_OBSNCHAN = "OBSNCHAN"
_HEADER_KEY_OBSBW = "OBSBW"
_HEADER_KEY_OBSFREQ = "OBSFREQ"
_HEADER_KEY_PKTIDX = "PKTIDX"
_HEADER_KEY_PKTSTART = "PKTSTART"
_HEADER_KEY_PKTSTOP = "PKTSTOP"
_HEADER_KEY_DIRECTIO = "DIRECTIO"

_HEADER_VALUE_SETIGEN = "SETIGEN"
_HEADER_VALUE_SYNTHETIC = "SYNTHETIC"
_HEADER_VALUE_SETIGEN_SUFFIX = "_SETIGEN"
_HEADER_VALUE_ENCODED_GUPPI_BACKEND = "'GUPPI   '"
_HEADER_VALUE_ENCODED_GBT_TELESCOPE = "'GBT     '"
_HEADER_VALUE_ENCODED_DEFAULT_OBSERVER = "'Dave MacMahon'"
_HEADER_VALUE_ENCODED_DEFAULT_SOURCE = "'TMC1    '"
_HEADER_VALUE_ENCODED_RAW_MODE = "'RAW     '"
_HEADER_VALUE_ENCODED_1SFA_FORMAT = "'1SFA    '"
_HEADER_VALUE_ENCODED_PREFIX = "'"
_HEADER_VALUE_DEFAULT_PACKET_INDEX = 0


_DEFAULT_HEADER_ENTRIES = (
    (_HEADER_KEY_BACKEND, _HEADER_VALUE_ENCODED_GUPPI_BACKEND),
    (_HEADER_KEY_TELESCOP, _HEADER_VALUE_ENCODED_GBT_TELESCOPE),
    (_HEADER_KEY_OBSERVER, _HEADER_VALUE_ENCODED_DEFAULT_OBSERVER),
    ("PROJID", "'AGBT20B_999_22'"),
    ("FRONTEND", "'RcvrArray18_26'"),
    ("NRCVR", "2"),
    ("FD_POLN", "'CIRC    '"),
    ("BMAJ", "0.009263915095687008"),
    ("BMIN", "0.009263915095687008"),
    (_HEADER_KEY_SRC_NAME, _HEADER_VALUE_ENCODED_DEFAULT_SOURCE),
    ("TRK_MODE", "'TRACK   '"),
    ("RA_STR", "'04:41:45.7920'"),
    ("RA", "70.4408"),
    ("DEC_STR", "'+25:41:27.9600'"),
    ("DEC", "25.6911"),
    ("LST", "83464"),
    ("AZ", "433.0963"),
    ("ZA", "69.1473"),
    ("DAQCTRL", "'start   '"),
    ("DAQPULSE", "'Tue Sep 22 00:24:27 2020'"),
    ("DAQSTATE", "'record  '"),
    (_HEADER_KEY_NBITS, "8"),
    ("OFFSET0", "0.0"),
    ("OFFSET1", "0.0"),
    ("OFFSET2", "0.0"),
    ("OFFSET3", "0.0"),
    ("BANKNAM", "'BLP00   '"),
    ("TFOLD", "0"),
    ("DS_FREQ", "1"),
    ("DS_TIME", "1"),
    ("FFTLEN", "512"),
    (_HEADER_KEY_CHAN_BW, "-2.9296875"),
    ("BANDNUM", "0"),
    ("NBIN", "0"),
    (_HEADER_KEY_OBSNCHAN, "64"),
    ("SCALE0", "1.0"),
    ("SCALE1", "1.0"),
    ("DATAHOST", "'blr2-1-10-0.gb.nrao.edu'"),
    ("SCALE3", "1.0"),
    (_HEADER_KEY_NPOL, "4"),
    ("POL_TYPE", "'AABBCRCI'"),
    ("BANKNUM", "0"),
    ("DATAPORT", "60000"),
    ("ONLY_I", "0"),
    ("CAL_DCYC", "0.5"),
    (_HEADER_KEY_DIRECTIO, "1"),
    (_HEADER_KEY_BLOCSIZE, "134217728"),
    ("ACC_LEN", "1"),
    ("CAL_MODE", "'OFF     '"),
    ("OVERLAP", "0"),
    ("OBS_MODE", _HEADER_VALUE_ENCODED_RAW_MODE),
    ("CAL_FREQ", "0.0"),
    ("DATADIR", "'/datax/dibas'"),
    (_HEADER_KEY_OBSFREQ, "25720.21484375"),
    ("PFB_OVER", "12"),
    (_HEADER_KEY_SCANLEN, "300.0"),
    ("PARFILE", "'/opt/dibas/etc/config/example.par'"),
    (_HEADER_KEY_OBSBW, "-187.5"),
    ("SCALE2", "1.0"),
    ("BINDHOST", "'eth4    '"),
    ("PKTFMT", _HEADER_VALUE_ENCODED_1SFA_FORMAT),
    (_HEADER_KEY_TBIN, "3.41333333333333E-07"),
    ("BASE_BW", "1450.0"),
    ("CHAN_DM", "0.0"),
    ("SCAN", "10"),
    ("STT_SMJD", "15868"),
    ("STT_IMJD", "59114"),
    ("STTVALID", "1"),
    ("NETSTAT", "'receiving'"),
    ("DISKSTAT", "'waiting '"),
    (_HEADER_KEY_PKTIDX, "0"),
    ("DROPAVG", "1.37455e-05"),
    ("DROPTOT", "0.6331"),
    ("DROPBLK", "0"),
    (_HEADER_KEY_PKTSTOP, "27459584"),
    ("NETBUFST", "'1/24    '"),
    ("STT_OFFS", "0"),
    ("SCANREM", "0.0"),
    ("PKTSIZE", "8192"),
    ("NPKT", "16384"),
    ("NDROP", "0"),
)


def _set_identity_header_value(header_dict: dict[str, Any],
                               *,
                               key: str,
                               fallback_value: Any,
                               input_header_dict: dict[str, Any] | None,
                               guard_value: str) -> None:
    """Set or normalize a human-readable identity field in the RAW header.

    Args:
        header_dict: Header dictionary being assembled.
        key: Header key to update.
        fallback_value: Default value when the key is absent.
        input_header_dict: Optional source header dictionary from an input RAW
            file.
        guard_value: Value fragment used to detect whether the field has
            already been marked as synthetic.
    """
    if key not in header_dict:
        header_dict[key] = fallback_value
    elif input_header_dict is not None and guard_value not in input_header_dict[key]:
        header_dict[key] = f"{input_header_dict[key].strip()}{_HEADER_VALUE_SETIGEN_SUFFIX}"


def _header_populate_configuration(backend: Any,
                                   header_dict: dict[str, Any] | None = None) -> dict[str, Any]:
    """Populate RAW header values derived from backend configuration.

    Args:
        backend: Raw-voltage backend providing runtime configuration.
        header_dict: Optional existing header overrides.

    Returns:
        Header dictionary populated with configuration-derived values.
    """
    header_dict = {} if header_dict is None else dict(header_dict)

    _set_identity_header_value(header_dict,
                               key=_HEADER_KEY_TELESCOP,
                               fallback_value=_HEADER_VALUE_SETIGEN,
                               input_header_dict=backend.input_header_dict,
                               guard_value=_HEADER_VALUE_SETIGEN)
    _set_identity_header_value(header_dict,
                               key=_HEADER_KEY_OBSERVER,
                               fallback_value=_HEADER_VALUE_SETIGEN,
                               input_header_dict=backend.input_header_dict,
                               guard_value=_HEADER_VALUE_SETIGEN)
    _set_identity_header_value(header_dict,
                               key=_HEADER_KEY_SRC_NAME,
                               fallback_value=_HEADER_VALUE_SYNTHETIC,
                               input_header_dict=backend.input_header_dict,
                               guard_value=_HEADER_VALUE_SYNTHETIC)

    header_dict[_HEADER_KEY_NBITS] = backend.num_bits
    header_dict[_HEADER_KEY_CHAN_BW] = backend.chan_bw * 1e-6
    header_dict[_HEADER_KEY_NPOL] = backend.num_pols

    header_dict[_HEADER_KEY_BLOCSIZE] = backend.block_size
    header_dict[_HEADER_KEY_SCANLEN] = backend.obs_length
    header_dict[_HEADER_KEY_TBIN] = backend.tbin
    if backend.is_antenna_array:
        header_dict[_HEADER_KEY_NANTS] = backend.num_antennas
    header_dict[_HEADER_KEY_OBSNCHAN] = backend.num_chans * backend.num_antennas
    header_dict[_HEADER_KEY_OBSBW] = backend.chan_bw * backend.num_chans * 1e-6

    center_freq = (backend.start_chan + (backend.num_chans - 1) / 2) * backend.chan_bw
    center_freq += backend.fch1
    header_dict[_HEADER_KEY_OBSFREQ] = center_freq * 1e-6

    if _HEADER_KEY_PKTIDX not in header_dict:
        header_dict[_HEADER_KEY_PKTIDX] = _HEADER_VALUE_DEFAULT_PACKET_INDEX
    header_dict[_HEADER_KEY_PKTIDX] = int(header_dict[_HEADER_KEY_PKTIDX])
    if _HEADER_KEY_PKTSTART not in header_dict:
        header_dict[_HEADER_KEY_PKTSTART] = header_dict[_HEADER_KEY_PKTIDX]
    header_dict[_HEADER_KEY_PKTSTOP] = int(header_dict[_HEADER_KEY_PKTSTART]) + backend.num_blocks * backend.samples_per_block

    return header_dict


def _header_add_from_template(header_dict: dict[str, Any] | None = None) -> dict[str, Any]:
    """Fill missing header values from the built-in default template.

    Args:
        header_dict: Optional existing header overrides.

    Returns:
        Header dictionary with default entries populated.
    """
    header_dict = {} if header_dict is None else dict(header_dict)

    for key, value in _DEFAULT_HEADER_ENTRIES:
        if key not in header_dict:
            header_dict[key] = value
    return header_dict


def _header_add_from_input_header(input_header_dict: dict[str, Any],
                                  header_dict: dict[str, Any] | None = None) -> dict[str, Any]:
    """Copy missing values from an input RAW header dictionary.

    Args:
        input_header_dict: Parsed input RAW header values.
        header_dict: Optional existing header overrides.

    Returns:
        Header dictionary with missing values populated from the input header.
    """
    header_dict = {} if header_dict is None else dict(header_dict)

    for key, value in input_header_dict.items():
        if key not in header_dict:
            header_dict[key] = value.strip()
    return header_dict


def _make_header(backend: Any, f: BinaryIO, header_dict: dict[str, Any]) -> None:
    """Write one RAW header block to an open file handle.

    Args:
        backend: Raw-voltage backend providing runtime configuration.
        f: Writable binary file handle.
        header_dict: Header values to serialize.
    """
    directio = False

    if _HEADER_KEY_DIRECTIO in header_dict:
        directio = header_dict[_HEADER_KEY_DIRECTIO]
        try:
            if isinstance(directio, str):
                directio = int(directio.replace(_HEADER_VALUE_ENCODED_PREFIX, ""))
            directio = directio != 0
        except BaseException as err:
            tqdm(f'Could not parse DIRECTIO value `{header_dict[_HEADER_KEY_DIRECTIO]}` ({repr(err)}). Replacing with `0`.')
            header_dict[_HEADER_KEY_DIRECTIO] = 0

    header_lines = 0
    for key, value in header_dict.items():
        value_is_encoded = isinstance(value, str) and value[0] == _HEADER_VALUE_ENCODED_PREFIX
        line = raw_utils.format_header_line(key, value, as_strings=value_is_encoded)
        f.write(f"{line:<80}".encode())
        header_lines += 1
    f.write(f"{'END':<80}".encode())
    header_lines += 1

    if directio:
        f.write(bytearray(512 - (80 * header_lines % 512)))

    header_dict[_HEADER_KEY_PKTIDX] += backend.samples_per_block


def _read_next_block(backend: Any) -> np.ndarray:
    """Read and decode the next RAW block from an input file.

    Args:
        backend: Raw-voltage backend configured for input-RAW passthrough.

    Returns:
        Complex voltage buffer for the next RAW block.

    Raises:
        ValueError: If the input RAW bit depth is unsupported.
    """
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
