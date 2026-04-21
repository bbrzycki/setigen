from __future__ import annotations

from pathlib import Path
from typing import Any

import click

from .reduction import RawReductionSpec, reduce_raw


def _validate_output_path(ctx: click.Context,
                          param: click.Parameter,
                          value: str | None) -> Path | None:
    """Convert an optional click path argument into a `Path`.

    Args:
        ctx: Active click context.
        param: Click parameter definition.
        value: Raw path value from click.

    Returns:
            `Path` instance or `None` when no value was supplied.
    """
    if value is None:
        return None
    return Path(value)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("input_path", type=click.Path(path_type=Path))
@click.argument("output_path", type=click.Path(path_type=Path))
@click.option("--fftlength",
              type=click.IntRange(min=1),
              required=True,
              help="Fine-channel FFT length.")
@click.option("--integration-factor",
              type=click.IntRange(min=1),
              required=True,
              help="Number of spectra to integrate in time.")
@click.option("--pol-mode",
              type=click.Choice(["1", "4", "-4"]),
              default="1",
              show_default=True,
              help="Polarization output mode: 1=Stokes I, 4=full-pol, -4=full Stokes.")
@click.option("--format",
              "output_format",
              type=click.Choice(["fil", "h5"]),
              required=True,
              help="Output filterbank format.")
@click.option("--start-chan",
              type=click.IntRange(min=0),
              default=None,
              help="First coarse channel to reduce.")
@click.option("--num-chans",
              type=click.IntRange(min=1),
              default=None,
              help="Number of coarse channels to reduce.")
@click.option("--backend",
              type=click.Choice(["auto", "numpy", "cupy"]),
              default="auto",
              show_default=True,
              help="Array backend for fine channelization.")
@click.option("--overwrite",
              is_flag=True,
              help="Allow overwriting the output file if it already exists.")
@click.option("--tmp-dir",
              callback=_validate_output_path,
              default=None,
              help="Optional temporary directory for staged writes.")
def main(input_path: Path,
         output_path: Path,
         fftlength: int,
         integration_factor: int,
         pol_mode: str,
         output_format: str,
         start_chan: int | None,
         num_chans: int | None,
         backend: str,
         overwrite: bool,
         tmp_dir: Path | None) -> None:
    """
    Reduce a GUPPI RAW file or RAW stem to a .fil or .h5 filterbank product.

    INPUT_PATH may be a specific *.0000.raw file or a RAW stem.
    OUTPUT_PATH is the destination .fil or .h5 file.

    Args:
        input_path: RAW stem or specific `.raw` file path.
        output_path: Destination `.fil` or `.h5` path.
        fftlength: Fine-channel FFT length.
        integration_factor: Number of spectra to integrate in time.
        pol_mode: Polarization output mode.
        output_format: Output file format.
        start_chan: First coarse channel to reduce.
        num_chans: Number of coarse channels to reduce.
        backend: Numerical array backend name.
        overwrite: Whether an existing output file may be replaced.
        tmp_dir: Optional directory for staged writes.
    """
    spec = RawReductionSpec(
        fftlength=fftlength,
        integration_factor=integration_factor,
        pol_mode=int(pol_mode),
        output_format=output_format,
        start_chan=start_chan,
        num_chans=num_chans,
        backend=backend,
    )
    final_path = reduce_raw(input_path,
                            output_path,
                            spec,
                            overwrite=overwrite,
                            tmp_dir=tmp_dir)
    click.echo(str(final_path))


if __name__ == "__main__":
    main()
