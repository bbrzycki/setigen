from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import glob
import math

from .. import raw_utils


@dataclass(frozen=True)
class _RawInputSpec:
    stem: Path
    files: tuple[Path, ...]
    header: dict
    header_size: int
    num_bits: int
    chan_bw: float
    ascending: bool
    num_pols: int
    num_antennas: int
    block_size: int
    obs_length: float
    tbin: float
    num_chans: int
    fch1: float
    source_name: str
    rawdatafile: str
    tstart_mjd: float


def _compute_header_size(header):
    return int(512 * math.ceil((80 * (len(header) + 1)) / 512))


def _normalize_num_pols(header_npol):
    num_pols = int(header_npol)
    if num_pols == 4:
        return 2
    return num_pols


def _resolve_tstart_mjd(header):
    try:
        stt_imjd = float(header["STT_IMJD"])
        stt_smjd = float(header.get("STT_SMJD", 0))
        stt_offs = float(header.get("STT_OFFS", 0))
    except KeyError:
        return 0.0
    return stt_imjd + (stt_smjd + stt_offs) / 86400.0


def _resolve_raw_files(input_path):
    input_path = Path(input_path)
    if input_path.suffix == ".raw":
        stem = raw_utils.get_stem(str(input_path))
    else:
        stem = input_path

    matches = sorted(glob.glob(f"{stem}.????.raw"))
    files = tuple(Path(match) for match in matches)
    if not files and input_path.suffix == ".raw" and input_path.exists():
        files = (input_path,)
        stem = raw_utils.get_stem(str(input_path))
    if not files:
        raise FileNotFoundError(f"No RAW files found for input path '{input_path}'.")
    return Path(stem), files


def _resolve_raw_input(input_path):
    stem, files = _resolve_raw_files(input_path)
    header = raw_utils.read_header(str(files[0]))
    header_size = _compute_header_size(header)

    num_bits = int(header["NBITS"])
    if num_bits not in (4, 8):
        raise ValueError(f"Unsupported RAW bit width: {num_bits}. Only 4-bit and 8-bit inputs are supported.")

    chan_bw = float(header["CHAN_BW"]) * 1e6
    ascending = chan_bw > 0
    num_antennas = int(header.get("NANTS", 1))
    if num_antennas != 1:
        raise NotImplementedError("RAW reduction currently supports single-antenna inputs only.")

    num_pols = _normalize_num_pols(header["NPOL"])
    if num_pols not in (1, 2):
        raise ValueError(f"Unsupported RAW polarization count: {num_pols}.")

    block_size = int(header["BLOCSIZE"])
    obs_length = float(header["SCANLEN"])
    tbin = float(header["TBIN"])
    num_chans = int(header["OBSNCHAN"]) // num_antennas
    center_freq = float(header["OBSFREQ"]) * 1e6
    fch1 = center_freq - ((num_chans - 1) / 2) * chan_bw

    source_name = header.get("SRC_NAME", "SYNTHETIC").strip().strip("'")
    rawdatafile = files[0].name
    tstart_mjd = _resolve_tstart_mjd(header)

    return _RawInputSpec(
        stem=stem,
        files=files,
        header=header,
        header_size=header_size,
        num_bits=num_bits,
        chan_bw=chan_bw,
        ascending=ascending,
        num_pols=num_pols,
        num_antennas=num_antennas,
        block_size=block_size,
        obs_length=obs_length,
        tbin=tbin,
        num_chans=num_chans,
        fch1=fch1,
        source_name=source_name,
        rawdatafile=rawdatafile,
        tstart_mjd=tstart_mjd,
    )


def _iter_raw_data_blocks(input_spec: _RawInputSpec, *, max_blocks=None):
    blocks_seen = 0
    for path in input_spec.files:
        with open(path, "rb") as handle:
            while True:
                header_bytes = handle.read(input_spec.header_size)
                if not header_bytes:
                    break
                data_chunk = handle.read(input_spec.block_size)
                if len(data_chunk) < input_spec.block_size:
                    break
                yield data_chunk
                blocks_seen += 1
                if max_blocks is not None and blocks_seen >= max_blocks:
                    return
