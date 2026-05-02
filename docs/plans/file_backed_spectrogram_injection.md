# File-Backed Spectrogram Injection Plan

This plan tracks the proposed replacement for `blimpy`-centered waterfall I/O
when injecting synthetic narrowband signals into large observational
spectrograms. The goal is to keep the current `Frame` API usable while allowing
read/write operations against many-GB `.fil` and `.h5` products without loading
the full observation into memory.

## Current Branch Context

The branch `feature/20260426-overhaul-I/O-towards-efficient-injection` already
moves `Frame` closer to this model:

- `Frame` owns serializable observation context instead of live waterfall file
  handles.
- Loading a waterfall copies data and header metadata, then closes file handles
  opened by `setigen`.
- Derived frames preserve observation context deliberately, including start
  time, source name, filterbank header, and user metadata.
- Axis helpers expose frequency/time centers and edges so plotting, injection,
  and drift-rate calculations can share one convention.

The current helper surfaces are:

- `setigen._frame.context._copy_frame_context()` for derived-frame metadata and
  header propagation.
- `setigen._frame.io` for waterfall adapter creation, serialization, and
  header/data synchronization.
- `setigen.voltage._reduction.frame_context._frame_context_kwargs()` for
  carrying filterbank context from voltage reductions into `Frame.from_data()`.

These were useful groundwork for the file-backed implementation slice below.

## Implemented Slice

The first implementation slice now exists on this branch:

- `Frame.open(path, mode="r")` opens `.h5`, `.hdf5`, and `.fil` products as
  read-only file-backed frames.
- `Frame.open(path, mode="r+", allow_inplace=True)` enables explicit direct
  mutation of an existing file.
- `Frame.open_copy(input_path, output_path)` copies the input file on disk and
  opens the copy as a writable file-backed frame. This is the preferred safe
  mutation path.
- File-backed frames are context managers and close their HDF5/raw file handles
  on exit.
- `Frame.read_frame(...)` reads a bounded region into a normal eager `Frame`.
  It accepts physical `f_range`/`t_range` or half-open
  `f_index_range`/`t_index_range`.
- `Frame.add_signal(...)` on a writable file-backed frame patches the backing
  file immediately in time chunks. It returns a `FileBackedSignalResult`
  summary instead of a full signal array, because returning a full array would
  defeat the memory-savings goal.
- `Frame.plot(...)` accepts the same region arguments and reads plotted
  file-backed data from the current backing store, so plots after injection show
  the patched signal.
- `NoiseEstimationConfig` and `NoiseStats` provide a shared sigma-clipping
  noise-stat path. Eager frames keep their existing full-frame behavior, while
  file-backed frames can estimate stats from explicit local context windows.
- `Frame.get_intensity(..., noise_stats=...)` and
  `Frame.get_snr(..., noise_stats=...)` can use explicit stats without changing
  legacy calls that rely on cached frame noise.

The implemented private modules are:

- `setigen._spectrogram.base`
- `setigen._spectrogram.h5`
- `setigen._spectrogram.fil`
- `setigen._frame.file_mutation`
- `setigen.noise`

Current limitations:

- File-backed synthetic noise mutation is intentionally not implemented. It
  would require broad full-file mutation, not narrowband chunk patching.
- `open_copy()` performs a full file copy up front. It is memory-bounded but
  still pays full-file disk I/O.
- SNR-based file-backed injection remains future work, but local file-backed
  noise-stat estimation is available for explicit contexts.
- File-backed `add_signal()` evaluates callable time/path profiles over the
  full time axis first so chunked rendering stays consistent across chunks.
  This is much smaller than loading the full spectrogram, but it is still an
  O(`tchans`) allocation.

## Goals

- Support `.fil` and `.h5` observations whose full spectrogram arrays are too
  large to fit comfortably in memory.
- Preserve the existing eager in-memory behavior:

  ```python
  frame = stg.Frame(waterfall="obs.h5")
  frame.add_signal(...)
  frame.save_hdf5("obs_injected.h5")
  ```

- Add an explicit file-backed API that still feels like `Frame`.
- Make mutating methods write immediately when called. There should be no lazy
  pending-injection plan that only materializes during a later save.
- Make copy-backed mutation the default safe path for real observations.
- Allow direct in-place mutation only through an explicit power-user API.
- Ensure `frame.plot()` after `frame.add_signal()` reads the patched backing
  file and shows the injected signal.

## Non-Goals

- Do not preserve `blimpy.Waterfall` as the internal abstraction for new
  file-backed writes.
- Do not hide large-file behavior behind sentinel arrays or partially loaded
  `.data` objects.
- Do not make SNR-based injection silently scan full multi-GB products unless
  the caller explicitly asks for that cost.
- Do not mutate original observational products by default.

## Blimpy Audit

`blimpy.Waterfall` is a facade over `.fil` and `.h5` readers. It parses headers,
normalizes selection ranges, optionally materializes data into `Waterfall.data`,
and exposes plotting plus format conversion helpers.

Useful behavior to keep:

- `.fil` reading computes the SIGPROC header length, converts a frequency range
  to channel indices, seeks row-by-row, and reads selected frequency spans.
- `.h5` reading uses HDF5 dataset slicing for hyperslab reads.
- Heavy-file conversion writes large products in blobs instead of materializing
  the entire observation.

Problems to avoid:

- Reader, writer, plotting, selection state, memory policy, and data ownership
  are mixed into one object.
- Large-file state is implicit rather than represented by a clear file-backed
  backend.
- Resource ownership is implicit; HDF5 handles can live longer than expected.
- Some writer failures use process exits, which is not acceptable library
  behavior.
- The public API is conversion-oriented, not injection-oriented. There is no
  clean mutation-by-region API for patching an affected window and returning.

## Proposed API

Existing eager behavior remains valid:

```python
frame = stg.Frame(waterfall="obs.h5")
frame.add_signal(...)
frame.save_hdf5("obs_injected.h5")
```

Read-only file-backed access:

```python
with stg.Frame.open("obs.fil", mode="r") as frame:
    small = frame.read_frame(f_range=(6000.0, 6000.1), t_range=(0, 32))
    frame.plot(f_range=(6000.0, 6000.1), t_range=(0, 128))
```

Safe writable copy-backed access:

```python
with stg.Frame.open_copy("obs.h5", "obs_injected.h5") as frame:
    frame.add_signal(..., auto_bounding=True, truncate_below=1e-3)
    frame.plot(f_range=(6000.0, 6000.1), t_range=(0, 128))
```

Explicit direct mutation:

```python
with stg.Frame.open("obs_copy.h5", mode="r+", allow_inplace=True) as frame:
    frame.add_signal(...)
```

`Frame.open_copy()` should be the preferred API for real observations. Phase 1
can copy the full file up front, which is disk-I/O heavy but memory-bounded.
Later phases can optimize this with staged sparse or region-copy strategies
where the file format makes that safe.

## Object Model

`Frame` should be backed by a small internal backend interface:

- `InMemoryFrameBackend`: owns a normal NumPy array and supports the existing
  implementation path.
- `FileFrameBackend`: owns a context-managed file backend, exposes metadata and
  region reads, and optionally exposes region writes.

Candidate private modules:

- `setigen._spectrogram.base`: shared metadata model, selection normalization,
  axis conversions, and the `SpectrogramBackend` protocol.
- `setigen._spectrogram.fil`: native SIGPROC `.fil` parser, row/region reader,
  staged writer, and optional `np.memmap` support for safe contiguous cases.
- `setigen._spectrogram.h5`: native HDF5 reader/writer using `h5py` datasets and
  hyperslab selection.
- `setigen._frame.backends`: in-memory and file-backed frame backend adapters.
- `setigen._frame.streaming_signal`: chunked signal rendering and immediate
  patch application.
- `setigen._frame.file_mutation`: copy guards, writable-region patching, and
  injection metadata bookkeeping.

The backend boundary should be private. Public users should still think in
terms of `Frame`.

## Mutable Write Semantics

A file-backed frame has either a read-only backend or a writable backend.

- `Frame.open(path, mode="r")` opens metadata and region readers only.
  `read_frame()` and `plot()` work, but `add_signal()` raises.
- `Frame.open_copy(input_path, output_path)` creates a writable product from
  the input observation. Mutating methods patch that writable product
  immediately.
- `Frame.open(path, mode="r+", allow_inplace=True)` opens an existing product
  for direct mutation. This should remain explicit because it can corrupt or
  alter original observational data.
- `add_signal()` on a writable file-backed frame resolves the affected region,
  processes that region in memory-bounded chunks, writes each patched chunk back
  before returning, and records injection metadata in memory and, where
  possible, in output metadata.
- `plot()` after `add_signal()` reads from the already-patched backing file.
  The plot should show the signal without a separate save step.

The important distinction is that the copy operation and the mutation operation
are separate safety concepts. `open_copy()` protects observational input.
Chunked mutation protects memory usage.

## Immediate Chunked Injection Algorithm

When `add_signal()` is called on a writable file-backed frame:

1. Validate that the backend is writable.
2. Resolve the injection's affected frequency range from explicit
   `bounding_f_range` or the existing `auto_bounding` machinery. If no safe
   range can be inferred, fall back to the full frequency span while still
   chunking over time.
3. Convert the affected frequency range into channel indices, respecting
   negative `foff` and ascending/descending channel order.
4. Choose a chunk shape from a memory budget, dtype, and file layout. Phase 1
   can process time-contiguous chunks across the affected frequency span.
5. For each chunk, read only `time[t0:t1]` and `freq[f0:f1]` from the backing
   file into a NumPy array.
6. Build a lightweight frame-like view for that chunk with correct `df`, `dt`,
   `fch1`, `ascending`, `t_start`, and `t_offset`.
7. Render the same signal profile that eager `Frame.add_signal()` would render,
   but only for this chunk's time/frequency window.
8. Add the rendered signal into the chunk array.
9. Write that chunk back to the same region of the writable backing file before
   moving to the next chunk.
10. Flush the backend and update injection metadata before `add_signal()`
    returns.

This makes the function call itself the point of mutation. There is no delayed
render/save stage.

## Why This Saves Memory

The eager path needs memory proportional to the selected observation:

```text
memory ~= n_times * n_chans * bytes_per_sample
```

The file-backed path needs memory proportional to the active chunk:

```text
memory ~= chunk_n_times * affected_n_chans * bytes_per_sample
```

For a narrowband injection, `affected_n_chans` can be a tiny fraction of the
full observation. A 24-hour product with millions of channels may be tens or
hundreds of GB, but a signal that occupies a few thousand channels can be
patched in chunks that are tens or hundreds of MB.

This also limits compute to the affected region. The signal renderer does not
need to evaluate profiles across the full spectrogram when the bounding region
is known.

`open_copy()` may still perform a full-file disk copy in phase 1. That is an
I/O cost, not a memory cost. The copy protects the original file; the chunked
patching protects memory.

## Backend Write Details

HDF5 writes should use dataset hyperslabs:

```python
dataset[t0:t1, if_id, f0:f1] = patched_chunk
```

The actual dimension order should be normalized behind the HDF5 backend so
`Frame` code does not know whether the on-disk data are shaped as
`(time, if, frequency)` or something backend-specific.

Filterbank writes are row-window overwrites. For a single IF and row-major
SIGPROC layout, the byte offset for a row window is:

```text
offset = header_size + ((time_index * nifs + if_id) * nchans + f0) * bytes_per_sample
```

Then the backend seeks to `offset` and writes `f1 - f0` samples for that time
row. Multi-IF products and frequency orientation should be handled inside the
backend.

Both backends need:

- metadata-only open
- context-manager close behavior
- region read
- optional region write
- explicit flush
- errors instead of process exits

## Plot Compatibility

Plotting should not require a separate API. For a file-backed frame,
`Frame.plot()` can request the displayed time/frequency window from
`FileFrameBackend.read_region()` and then use the same plotting code path as an
eager frame.

That means:

- read-only file-backed plots show current on-disk data
- writable file-backed plots show patched data after `add_signal()`
- very large plots should require an explicit visible region or downsampling
  strategy rather than accidentally loading an entire observation

## Noise and SNR Policy

Explicit signal `level` can work immediately for file-backed injection.

`get_intensity(snr=...)` currently depends on `Frame.noise_std`, which assumes
materialized data. File-backed frames need an explicit estimator before
SNR-based injection is reliable:

```python
stats = frame.estimate_noise_stats(
    path=stg.constant_path(...),
    f_profile=stg.box_f_profile(...),
    t_range=(0, 1024),
    config=stg.NoiseEstimationConfig(
        context_width=2048,
        guard_width=64,
    ),
)
```

Exact full-file sigma clipping should not be the default for GB-scale files, and
it is not scientifically right for many BL-style products. PFB scalloping and
coarse-channel bandpass structure mean the local baseline and variance can
change across frequency. A global mean/std can overstate or understate the
true local SNR depending on where the injected signal lands in the spectral
response.

The first SNR implementation should use local context windows:

- Resolve the signal path and its affected channels using the same bounding
  machinery used for injection.
- For each time chunk, read a surrounding frequency window around the signal
  path, not the full observation.
- Exclude the signal-support region plus a guard band before estimating noise.
  This matters both before injection and after injection if users ask to inspect
  achieved SNR.
- Estimate background with local sigma clipping by default, matching the
  existing `Frame._update_noise_frame_stats()` convention. The initial defaults
  should stay consistent with the current eager path: `sigma=3`, `maxiters=5`,
  and unmasked output before computing mean/std.
- Keep median/MAD as an alternate robust estimator, but do not make it the
  default unless we intentionally decide to change setigen's broader noise
  convention.
- Allow users to choose the context scale in channels or Hz. The default should
  be wide enough to sample nearby bandpass/scalloping behavior but narrow enough
  not to cross unrelated coarse-channel structure unnecessarily.
- Return structured metadata: local mean, local std, number of samples, context
  bounds, excluded bounds, method, and whether the estimate was sampled or
  exhaustive.

Candidate API:

```python
config = stg.NoiseEstimationConfig(
    method="sigma_clip",
    sigma=3,
    maxiters=5,
    context_width=2048,
    guard_width=64,
    width_unit="channels",
)

stats = frame.estimate_noise_stats(
    path=path,
    bounding_f_range=None,
    auto_bounding=True,
    t_range=(0, 1024),
    config=config,
)

level = frame.get_intensity(snr=25, noise_stats=stats)
```

The config object should be a small typed public spec, probably a dataclass or
frozen model, rather than a loose dictionary. That gives us stable defaults,
validation, serialization into injection metadata, and an obvious extension
point for later response-aware estimation. A possible shape:

```python
@dataclass(frozen=True)
class NoiseEstimationConfig:
    method: str = "sigma_clip"
    sigma: float = 3
    maxiters: int = 5
    context_width: int | float = 2048
    guard_width: int | float = 64
    width_unit: str = "channels"
    combine: str = "pooled"
```

`combine="pooled"` would compute one effective mean/std from all accepted local
samples. Later options might preserve per-chunk statistics or weight chunks by
the expected spectral response.

The estimator should mirror `Frame` semantics without freezing current `Frame`
internals as perfect. The public meaning of `get_noise_stats()`,
`get_intensity(snr=...)`, and `get_snr(...)` should stay stable, while the
shared implementation becomes more explicit and better documented. Eager
`Frame` should keep full-frame sigma clipping as its default for backward
compatibility. File-backed frames should use the same estimator over local
context windows when SNR is requested for an injection path.

For file-backed `add_signal(snr=...)` support, avoid implicit full-file
estimation. Require either caller-supplied `noise_stats` or enough local-window
configuration to make the estimator explicit and reproducible.

Longer term, `estimate_noise_stats()` can support a path-local mode that walks
the signal path in chunks, estimates a local baseline per time/frequency
neighborhood, and combines those local estimates into one effective intensity
scale for the requested injection.

## Spectral Response Localization

A separate but related analysis goal is to infer where a file appears to sit in
the instrumental spectral response. We usually know where the PFB/coarse-channel
response should be, but real products may have slight offsets or calibration
imperfections.

This should be treated as analysis metadata, not a prerequisite for the first
file-backed injection implementation. Possible long-term approach:

- Estimate a smooth or periodic bandpass response over local frequency windows.
- Compare measured response shape against expected PFB/coarse-channel response.
- Fit a small phase/offset parameter indicating where fine channels appear to
  lie in the scalloping pattern.
- Cache that response model on the file-backed frame and expose it to SNR
  estimation so local noise windows can be interpreted in instrumental context.
- Keep this opt-in and inspectable; do not silently correct injected amplitudes
  using an inferred response model until the science assumptions are validated.

This could eventually support APIs such as:

```python
response = frame.estimate_spectral_response(
    f_range=(...),
    t_range=(...),
    model="pfb_scalloping",
)

stats = frame.estimate_noise_stats(path=path, response_model=response)
```

## Frame Improvement Roadmap

The file-backed work exposes a few `Frame` improvements that are useful even
without large files:

1. Done: extract the current sigma-clipped noise calculation into a shared
   `NoiseEstimationConfig` / `NoiseStats` implementation. This keeps eager
   behavior compatible while making the science assumptions explicit.
2. Split signal rendering from mutation. `Frame.add_signal()` currently renders
   and mutates in one method. A reusable chunk renderer would let eager,
   file-backed, and GPU paths share exactly the same signal model.
3. Add optional CuPy acceleration after the renderer split. CuPy should be an
   optional backend, not a hard dependency. The file-backed path can read a
   NumPy chunk, transfer it to GPU, render and add the signal, transfer the
   patched chunk back, and write it. The package-wide policy for this lives in
   `docs/plans/array_backend_acceleration.md`.
4. Make built-in paths and profiles backend-aware where practical. User
   callables in GPU mode may need to be CuPy-compatible or explicitly fall back
   to CPU rendering.
5. Add a clear dtype policy for rendering and writing. Compute dtype and output
   dtype should be explicit, especially for HDF5/FIL products.
6. Add injection metadata/history including signal parameters, noise config and
   stats, chunking, and CPU/GPU backend.
7. Revisit `frame.data` for file-backed frames. It currently materializes the
   full file for compatibility, but long-term large-file workflows should prefer
   explicit `read_frame()` or `materialize()` calls with size guards.

## Implementation Phases

1. Done: build read-only metadata and region readers for `.fil` and `.h5`, with
   context-manager ownership and tests against `blimpy` on small fixtures.
2. Done: add `Frame.open()` and `read_frame()` so users can choose eager or lazy
   behavior explicitly.
3. Done: add writable region backends and `Frame.open_copy()`. Start with
   full-file copy creation because it is simple, safe, and memory-bounded.
4. Done: make file-backed `add_signal()` patch writable chunks immediately and make
   `plot()` read back from the patched file.
5. Next: optimize narrowband injections by copying or reading unaffected spans
   directly instead of materializing entire rows or full time blocks.
6. Partly done: add local-window file-backed noise-stat estimation and SNR
   conveniences. Explicit stats are available; direct `add_signal(snr=...)`
   integration remains next.
7. Later: consider staged sparse output strategies after correctness and resource
   ownership are solid.
8. Later: add opt-in spectral-response localization for PFB scalloping and
   calibration-offset analysis.
9. Later: split signal rendering from mutation and add optional CuPy
   acceleration on top of the shared renderer. The acceleration work should
   follow the shared array backend spec in
   `docs/plans/array_backend_acceleration.md`.

## Validation Requirements

- Small `.fil` and `.h5` fixtures must produce value-equivalent output to eager
  `Frame` injection.
- Tests must assert memory stays bounded by configured chunk size.
- HDF5 file handles and raw file handles must close after normal and
  exceptional exits.
- Negative `foff`, ascending frequency products, `nifs`, and time/frequency
  sub-selections need direct coverage.
- Read-only file-backed `add_signal()` must raise a clear error.
- Copy-backed mutation must not alter the input path.
- Direct in-place update, if added, must be opt-in and separately tested.

## Open Decisions

- Should the public constructor be `Frame.open(...)`, `stg.open_frame(...)`, or
  both?
- Should direct mutation require both `mode="r+"` and `allow_inplace=True`, or
  is one explicit flag enough?
- What should the default memory budget be for chunked injection?
- What local context width and guard width should be the default for SNR
  estimation, and should widths be specified in channels, Hz, or both?
- Should local noise stats be summarized into one effective SNR scale, or should
  we preserve per-chunk/per-path statistics for downstream analysis?
- Should plotting an unbounded large file-backed frame raise, downsample, or ask
  for an explicit range?
- What injection metadata should be written into HDF5 attrs or filterbank
  headers versus kept only in the Python object?
