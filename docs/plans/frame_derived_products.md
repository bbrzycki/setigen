# Frame Derived Product Plan

This plan tracks the `Frame`, `Spectrum`, `TimeSeries`, and helper cleanup that
should happen alongside file-backed injection. These products are common
scientific operations, so they should have clear semantics, consistent metadata,
and memory-bounded behavior on large observations.

## Current State

The current branch has moved `Frame` in the right direction:

- Frame construction is split into `setigen._frame.construction`.
- Waterfall/header synchronization is split into `setigen._frame.io`.
- Signal rendering helpers live in `setigen._frame.signal`.
- File-backed mutation lives in `setigen._frame.file_mutation`.
- File-backed spectrogram readers live in `setigen._spectrogram`.
- Noise statistics now share `NoiseEstimationConfig` and `NoiseStats`.
- `slice`, `dedrift`, `read_frame`, and `integrate` use
  `_finalize_derived_frame()` so derived frames preserve source name, start
  time, headers, custom metadata, and operation provenance.
- Plot axis logic is centralized in `setigen._plot.axes`.
- Done: `Spectrum` and `TimeSeries` accept one-dimensional data and coerce it
  into the singleton-axis frame representation.
- Done: `read_frame`, `slice`, `dedrift`, `spectrum`, and `timeseries` attach
  explicit `metadata["derived"]` payloads and synchronize derived headers.
- Done: `spectrum()` and `timeseries()` support region selectors and chunked
  file-backed `mean` / `sum` reductions.

This is good groundwork. `Frame` is still doing too much, but the newly added
helpers are now strong enough to support incremental cleanup instead of a large
rewrite.

## Spectrum And TimeSeries Semantics

The current representation mostly makes sense:

- A `Spectrum` is a one-row `Frame` with shape `(1, fchans)`.
- A `TimeSeries` is a one-channel `Frame` with shape `(tchans, 1)`.
- `spectrum(frame, mode="mean" | "sum")` integrates over time.
- `timeseries(frame, mode="mean" | "sum")` integrates over frequency.

This model is worth keeping because it reuses existing axis, plotting, I/O, and
metadata behavior. The singleton-axis convention also serializes naturally into
filterbank/HDF5-style products.

The physical parameter interpretation should be explicit:

- `Spectrum.df` remains the source channel width.
- `Spectrum.dt` becomes the collapsed time span represented by the spectrum.
- `Spectrum.fch1` follows the selected frequency span.
- `Spectrum.t_start` remains the start time of the selected time span.
- `TimeSeries.dt` remains the source time resolution.
- `TimeSeries.df` becomes the collapsed frequency bandwidth.
- `TimeSeries.fch1` should be the center frequency of the collapsed frequency
  span, so its single channel has meaningful edges at
  `fch1 +/- df / 2`.
- `TimeSeries.t_start` follows the selected time span.

## Problems To Fix

1. Derived-product provenance is under-specified.

   `Spectrum` and `TimeSeries` currently inherit custom metadata, but they do
   not record that they were produced by integrating a source frame, which axis
   was collapsed, what reducer was used, what region was selected, or whether
   normalization was applied.

2. Header propagation is too literal.

   `_copy_frame_context()` copies `frame.header`, while the derived frame's
   core dimensions live separately in `fchans`, `tchans`, `df`, `dt`, and
   `fch1`. Save paths rebuild headers from the frame object, so serialization is
   mostly protected, but the in-memory `header` attribute can be stale after
   `slice`, `dedrift`, or `integrate`.

3. Done: `integrate()` used to materialize file-backed frames.

   `spectrum()` and `timeseries()` now reduce file-backed frames in chunks for
   `mean` and `sum`. Reducers beyond those still need explicit memory policy.

4. The reduction spec is too small.

   `IntegrationMode` currently supports `mean` and `sum`. Those are the right
   initial reducers, but common frame products also need a consistent place for
   selected time/frequency ranges, normalization policy, local noise settings,
   and future reducers.

5. Normalization vocabulary is inconsistent.

   `integrate(normalize=True)` applies sigma-clipped zero-mean/unit-variance
   normalization to the reduced output. `Spectrum.plot(snr=True)` does a
   sigma-normalized copy. `TimeSeries.plot(norm=True)` divides by the mean.
   These are all useful, but they should not share one vague word.

6. Direct `Spectrum` and `TimeSeries` constructors are not friendly to common
   one-dimensional inputs.

   The classes internally store singleton-axis 2D arrays, but direct users
   should be able to pass a 1D spectrum or time series and get the expected
   shape. Constructor guards should raise `ValueError`, not `assert`.

## Shared Collapse Spec

Add a shared collapse/reduction spec and route `integrate()`, `spectrum()`, and
`timeseries()` through it.

Candidate public surface:

```python
frame.spectrum(
    reducer="mean",
    normalize=None,
    f_range=None,
    t_range=None,
    f_index_range=None,
    t_index_range=None,
    noise_config=None,
    max_chunk_bytes=None,
)

frame.timeseries(
    reducer="mean",
    normalize=None,
    f_range=None,
    t_range=None,
    f_index_range=None,
    t_index_range=None,
    noise_config=None,
    max_chunk_bytes=None,
)
```

The existing top-level functions should remain:

```python
stg.spectrum(frame, mode="mean")
stg.timeseries(frame, mode="mean")
stg.integrate(frame, axis="time", mode="mean")
```

but internally they should normalize into a shared spec.

Initial reducer support:

- `mean`
- `sum`

Future reducer support can include `median`, `std`, `max`, and percentile-like
reducers, but only after we decide how to handle them in chunked file-backed
mode.

Initial normalization support:

- `None` or `"none"`: return physical integrated power.
- `"sigma_clip"`: use `NoiseEstimationConfig` on the reduced product.
- `"mean"`: divide by the reduced-product mean.

Avoid a bare `normalize=True` in new APIs. Keep it as a compatibility alias
where it already exists.

## Derived Metadata

Derived products should carry source context and explicit operation metadata.
Use plain Python primitives so metadata remains serializable.

Candidate structure:

```python
metadata["derived"] = {
    "operation": "integrate",
    "product_type": "spectrum",
    "collapsed_axis": "time",
    "reducer": "mean",
    "normalization": None,
    "source_shape": (tchans, fchans),
    "source_bounds": {
        "time_index_range": (t_start, t_stop),
        "frequency_index_range": (f_start, f_stop),
        "time_range_s": (time_start_s, time_stop_s),
        "frequency_range_hz": (f_min_hz, f_max_hz),
    },
}
```

For `TimeSeries`, use:

```python
metadata["derived"] = {
    "operation": "integrate",
    "product_type": "timeseries",
    "collapsed_axis": "frequency",
    ...
}
```

If normalization uses noise statistics, add:

```python
metadata["derived"]["noise_stats"] = noise_stats_metadata
metadata["derived"]["noise_config"] = noise_config_metadata
```

For `slice` and `dedrift`, use the same `derived` key with operation-specific
payloads instead of only copying free-form metadata.

## Header Policy

Create a helper that copies source context and updates frame-specific header
fields after a derived product is created.

Candidate private helper:

```python
_finalize_derived_frame(
    source,
    target,
    *,
    operation,
    product_type=None,
    source_bounds=None,
    reducer=None,
    normalization=None,
)
```

It should:

- deep-copy non-core custom metadata from the source
- attach `metadata["derived"]`
- deep-copy the source header when present
- update header fields that must reflect the target frame:
  `source_name`, `tsamp`, `tstart`, `nchans`, `nifs`, `fch1`, and `foff`

This should replace direct `_copy_frame_context()` calls over time.

## File-Backed Reduction

`spectrum()` and `timeseries()` should be memory-bounded for file-backed frames.

For `Spectrum`:

- select the requested frequency/time region
- read time chunks
- accumulate sum and count per frequency channel
- return sum or mean without loading the full observation

For `TimeSeries`:

- select the requested frequency/time region
- read time chunks and, if needed, frequency chunks
- reduce each time row across selected frequency channels
- return sum or mean without loading the full observation

This first file-backed implementation should support `mean` and `sum`. Reducers
such as `median` need either larger memory, temporary storage, or approximate
streaming algorithms and should be added deliberately.

## Implementation Order

1. Done: add constructor coercion for 1D `Spectrum` and `TimeSeries` inputs, replacing
   singleton-axis `assert` checks with `ValueError`.
2. Done: add a shared derived-frame finalizer that updates metadata and headers.
3. Done: route `slice`, `dedrift`, `read_frame`, `spectrum`, and `timeseries` through
   the finalizer.
4. Done: add operation metadata tests for spectrum, timeseries, slicing, and dedrift.
5. Done: add region-aware spectrum/timeseries APIs using the existing
   `f_range`/`t_range` and index-range conventions.
6. Done: add file-backed chunked `mean`/`sum` reductions.
7. Revisit normalization names and preserve current booleans as compatibility
   aliases.
8. Add docs examples showing eager and file-backed spectrum/timeseries
   workflows.
