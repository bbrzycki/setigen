# Architecture

This document summarizes the current structure of `setigen` and the boundaries
that should remain stable as the library evolves.

## Public Surface

The main public entry points are:

- `setigen.Frame`: synthetic or observational dynamic spectra
- `setigen.Cadence` and `setigen.OrderedCadence`: frame collections and
  observation ordering
- `setigen.Spectrum` and `setigen.TimeSeries`: integrated 1D views over a
  `Frame`-compatible data model
- `setigen.funcs`: reusable signal path, time-profile, frequency-profile, and
  bandpass-profile generators
- `setigen.voltage`: synthetic voltage streams, antenna models, channelization,
  requantization, and GUPPI RAW recording

Compatibility work should treat these public modules, classes, and user-facing
function signatures as the primary contract.

## Core Spectrogram Model

`setigen.Frame` is the central spectrogram abstraction.

Key responsibilities:

- represent waterfall-like intensity data with frequency and time metadata
- synthesize noise and injected signals
- load and save supported spectrogram formats
- provide plotting and convenience accessors used throughout the library

Internally, `Frame` delegates cohesive responsibilities to private modules
under `setigen._frame`:

- `construction.py`: initialization, precedence rules, and waterfall loading
- `models.py`: internal normalization of frame-side selector/config choices
- `signal.py`: signal construction and injection helpers
- `io.py`: serialization and waterfall export helpers

These modules are internal structure, not public API.

## Plotting Layer

Plotting remains user-facing through `Frame.plot()`, `Cadence.plot()`,
`Spectrum.plot()`, `TimeSeries.plot()`, and the top-level plot helpers.

Shared axis resolution and label/extent logic lives in `setigen._plot.axes`.
That private subsystem exists to keep axis semantics consistent across all
plotting surfaces without duplicating formatter logic.

## Cadence Layer

`Cadence` and `OrderedCadence` are lightweight coordination classes around
multiple `Frame` objects.

Their role is to:

- validate shape and axis compatibility across frames
- coordinate time offsets for multi-observation injections
- support cadence plotting and consolidation into a single frame

Cadence behavior should remain simple and predictable. It is intentionally not
a heavy orchestration framework.

## Voltage Layer

`setigen.voltage` models a lower-level synthetic observation pipeline.

Major public concepts:

- `DataStream` and `BackgroundDataStream`: voltage samples over time
- `Antenna` and `MultiAntennaArray`: polarization streams and multi-antenna
  coordination
- quantizers, polyphase filterbank helpers, and backend recording
- `RawVoltageBackend`: assembly and recording of GUPPI RAW output

Internal voltage structure is split into focused private packages:

- `setigen.voltage._antenna`: antenna construction and array-delay helpers
- `setigen.voltage._backend`: recording config, header handling, file
  orchestration, and backend pipeline steps

As with the frame helpers, these internal packages are implementation details.

## Assets and I/O Boundaries

The repository includes packaged assets under:

- `setigen/assets`
- `setigen/voltage/assets`

Runtime I/O should remain isolated to the parts of the library that actually
need it. Scientific modeling code should not take unnecessary dependencies on
file-format details.

## Testing Strategy

The test suite mixes broad regression coverage with focused public-surface
tests. When changing behavior:

- prefer adding direct tests near the touched feature
- preserve deterministic seeds and stable numerical assumptions
- add integration-style coverage when refactors cross subsystem boundaries

## Design Rules

- Favor small, cohesive modules over large files that mix unrelated concerns.
- Do not replace simple public methods with one-hop private wrappers.
- Use private subpackages when they represent a real subsystem boundary.
- Preserve `astropy.units` semantics, frequency-ordering behavior, and
  reproducibility guarantees across refactors.
