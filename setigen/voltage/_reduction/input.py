from __future__ import annotations

from dataclasses import dataclass
import glob
import math
from pathlib import Path
from typing import Any, Iterator

from .. import raw_utils


@dataclass(frozen=True)
class _RawInputSpec:
    """Normalized description of a RAW input stem."""

    stem: Path
    files: tuple[Path, ...]
    header: dict[str, Any]
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


def _compute_header_size(header: dict[str, Any]) -> int:
    """Compute padded RAW header size in bytes.

    Args:
        header: Parsed RAW header dictionary.

    Returns:
        Padded header size in bytes.
    """
    return int(512 * math.ceil((80 * (len(header) + 1)) / 512))


def _normalize_num_pols(header_npol: Any) -> int:
    """Normalize RAW `NPOL` values to the number of signal polarizations.

    Args:
        header_npol: Raw `NPOL` header value.

    Returns:
        Number of signal polarizations represented by the input.
    """
    num_pols = int(header_npol)
    if num_pols == 4:
        return 2
    return num_pols


def _resolve_tstart_mjd(header: dict[str, Any]) -> float:
    """Resolve observation start time in MJD from RAW header fields.

    Args:
        header: Parsed RAW header dictionary.

    Returns:
        Observation start time in MJD, or `0.0` when the header is missing the
        required timing fields.
    """
    try:
        stt_imjd = float(header["STT_IMJD"])
        stt_smjd = float(header.get("STT_SMJD", 0))
        stt_offs = float(header.get("STT_OFFS", 0))
    except KeyError:
        return 0.0
    return stt_imjd + (stt_smjd + stt_offs) / 86400.0


def _resolve_raw_files(input_path: str | Path) -> tuple[Path, tuple[Path, ...]]:
    """Resolve a RAW stem or `.raw` file into the concrete file sequence.

    Args:
        input_path: RAW stem or specific `.raw` file path.

    Returns:
        Tuple of normalized stem path and concrete RAW files.

    Raises:
        FileNotFoundError: If no matching RAW files can be found.
    """
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


def _resolve_raw_input(input_path: str | Path) -> _RawInputSpec:
    """Parse and validate the first RAW header for a reduction run.

    Args:
        input_path: RAW stem or specific `.raw` file path.

    Returns:
        Normalized RAW input description.

    Raises:
        ValueError: If the RAW file uses unsupported bit depth or polarization
            count.
        NotImplementedError: If the input uses a multi-antenna RAW format that
            this reducer does not yet support.
    """
    stem, files = _resolve_raw_files(input_path)
    header = raw_utils.read_header(str(files[0]))
    header_size = _compute_header_size(header)

    num_bits = int(header["NBITS"])
    if num_bits not in (4, 8):
        raise ValueError(
            f"Unsupported RAW bit width: {num_bits}. Only 4-bit and 8-bit inputs are supported."
        )

    chan_bw = float(header["CHAN_BW"]) * 1e6
    ascending = chan_bw > 0
    num_antennas = int(header.get("NANTS", 1))
    if num_antennas != 1:
        raise NotImplementedError(
            "RAW reduction currently supports single-antenna inputs only."
        )

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


def _iter_raw_data_blocks(
    input_spec: _RawInputSpec,
    *,
    max_blocks: int | None = None,
) -> Iterator[bytes]:
    """Yield RAW data payloads block by block across all files in a stem.

    Args:
        input_spec: Normalized RAW input description.
        max_blocks: Optional maximum number of blocks to yield.

    Yields:
        RAW data payloads, one block at a time.
    """
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
