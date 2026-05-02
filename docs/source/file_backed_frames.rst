.. |setigen| replace:: :mod:`setigen`

File-backed frames
==================

Large observational spectrograms are often many GBs. Loading the full dynamic
spectrum into memory just to inject a narrowband signal is wasteful, and for
some observations it is not practical at all. File-backed frames provide a
memory-bounded alternative while keeping the main :class:`setigen.Frame`
interface.

The important distinction is:

``stg.Frame(waterfall=...)``
    Eager loading. The selected waterfall data are copied into memory and file
    handles opened by |setigen| are closed after loading.

``stg.Frame.open(...)``
    File-backed loading. Metadata are loaded immediately, but spectrogram data
    are read from the backing file only when a method requests a region.

Use eager frames for small synthetic products, unit tests, and workflows that
need the whole array. Use file-backed frames for real observations where the
full spectrogram is too large to materialize casually.

Opening observations
--------------------

Read-only access is the safest way to inspect large files:

.. code-block:: python

    import setigen as stg

    with stg.Frame.open("observation.h5", mode="r") as frame:
        print(frame.shape)
        print(frame.df, frame.dt)
        small = frame.read_frame(f_index_range=(1000, 1200),
                                 t_index_range=(0, 64))

The returned ``small`` object is a normal eager :class:`setigen.Frame` containing
only the requested region.

The same range convention is used by ``read_frame()`` and ``plot()``:

.. code-block:: python

    with stg.Frame.open("observation.fil", mode="r") as frame:
        frame.plot(f_range=(6095.0e6, 6095.1e6),
                   t_range=(0, 300),
                   db=True)

Ranges are relative to the frame:

- ``f_range`` is a physical frequency range in Hz or Astropy frequency units.
- ``t_range`` is a relative time range in seconds or Astropy time units.
- ``f_index_range`` and ``t_index_range`` are half-open index ranges.

Safe mutation by copying
------------------------

The preferred way to inject into real observations is copy-backed mutation:

.. code-block:: python

    from astropy import units as u
    import setigen as stg

    with stg.Frame.open_copy("observation.h5",
                             "observation_injected.h5",
                             overwrite=True) as frame:
        result = frame.add_signal(
            path=stg.constant_path(f_start=6095.05e6,
                                   drift_rate=1.5 * u.Hz / u.s),
            t_profile=stg.constant_t_profile(level=10),
            f_profile=stg.gaussian_f_profile(width=40 * u.Hz),
            auto_bounding=True,
            truncate_below=1e-3,
        )
        frame.plot(f_range=(6095.04e6, 6095.06e6),
                   t_range=(0, 300))

``Frame.open_copy()`` copies the input file on disk, opens the copy as writable,
and leaves the original observation unchanged. Mutating methods patch the output
file immediately; there is no deferred injection plan that only takes effect on
save.

Direct in-place mutation is available, but it is intentionally explicit:

.. code-block:: python

    with stg.Frame.open("working_copy.h5",
                        mode="r+",
                        allow_inplace=True) as frame:
        frame.add_signal(...)

Only use direct mutation on a file that is already safe to modify.

Chunked signal injection
------------------------

For writable file-backed frames, :meth:`setigen.Frame.add_signal` processes the
affected frequency region in time chunks:

1. Resolve the frequency channels touched by the signal.
2. Read a bounded time/frequency chunk from the backing file.
3. Render the synthetic signal contribution for that chunk.
4. Add the contribution to the chunk.
5. Write the chunk back to the backing file before returning.

This keeps memory bounded by the configured chunk size and affected frequency
width, not by the full observation shape. The return value is a
``FileBackedSignalResult`` summary rather than a full signal array, because
returning a full signal array would defeat the purpose of file-backed injection.

The time chunk is the number of time bins processed in one read-render-write
step. You can set it directly with ``chunk_tchans``. Otherwise |setigen| chooses
it from ``max_chunk_bytes`` and the affected frequency width:

.. code-block:: python

    result = frame.add_signal(...,
                              auto_bounding=True,
                              max_chunk_bytes=64 * 1024**2)

The default memory budget is 256 MiB. Internally, |setigen| reserves space for
the data chunk and the temporary arrays used during signal rendering, so the
chosen time chunk is conservative. A narrow bounded signal can usually process
many time bins per chunk; an unbounded injection over the full band may need
small time chunks.

Behind the scenes, file-backed injection follows the same signal model as eager
injection, but applies it to a small frame-like chunk view:

1. Resolve ``auto_bounding`` or ``bounding_f_range`` into a frequency slice.
2. Evaluate callable path and time profiles once on the full time axis when
   needed, preserving stochastic consistency across chunks.
3. For each time chunk, read ``[t_start:t_stop, f_start:f_stop]`` from the
   backing file.
4. Render the signal on that chunk's local time/frequency axes using the shared
   frame rendering helpers.
5. Add the signal into the chunk data and write that region back immediately.
6. Flush the backend and return ``FileBackedSignalResult`` with the patched
   frequency slice, number of time chunks, and largest chunk shape.

Use ``auto_bounding=True`` with built-in finite-width profiles where possible.
For profiles with long tails, also provide a scientifically meaningful
``truncate_below`` value:

.. code-block:: python

    frame.add_signal(...,
                     auto_bounding=True,
                     truncate_below=1e-3)

If a custom profile has broad support, pass an explicit ``bounding_f_range`` so
the renderer does not need to evaluate the entire band.

Noise and SNR with local context
--------------------------------

For real observations, a global mean and standard deviation can be a poor SNR
reference. Bandpass structure, coarse-channel response, and PFB scalloping can
make the noise level vary across frequency. File-backed workflows therefore
prefer explicit local noise estimation.

Signal rendering bounds and noise-estimation context are separate concepts:

``signal render bounds``
    The smallest frequency region where injected signal power is non-negligible.
    These bounds control which file-backed data are patched.

``noise guard``
    A region around the signal support that is excluded from the background
    estimate. This keeps the target track, and nearby line-like contamination,
    from defining its own noise level.

``noise context``
    A wider local region around the guarded signal support used to estimate the
    background mean and standard deviation.

This distinction matters for thin, low-drift narrowband signals. Such a signal
may only require a few tens of channels for mutation, but it should usually use
thousands of neighboring channels for SNR calibration. Tight render bounds save
I/O; wide local context gives a statistically useful background estimate.

Use :class:`setigen.NoiseEstimationConfig` to describe the local context:

.. code-block:: python

    from astropy import units as u
    import setigen as stg

    config = stg.NoiseEstimationConfig(
        method="sigma_clip",
        sigma=3,
        maxiters=5,
        context_width=2048,
        guard_width=64,
        width_unit="channels",
    )

    with stg.Frame.open("observation.h5", mode="r") as frame:
        stats = frame.estimate_noise_stats(
            path=stg.constant_path(f_start=6095.05e6,
                                   drift_rate=1.5 * u.Hz / u.s),
            f_profile=stg.gaussian_f_profile(width=40 * u.Hz),
            auto_bounding=True,
            truncate_below=1e-3,
            t_range=(0, 300),
            config=config,
        )
        level = frame.get_intensity(snr=20, noise_stats=stats)

The guard region excludes the signal neighborhood from the noise sample. The
context region supplies nearby channels used for the estimate. Sigma clipping
is the default because it matches the existing frame behavior while allowing
bright outliers to be rejected.

With ``path`` and ``f_profile`` supplied, ``estimate_noise_stats()`` first
resolves the signal support. It then reads a rectangular local context:

.. code-block:: text

    [min_signal_channel - context_width,
     max_signal_channel + context_width]

and excludes:

.. code-block:: text

    [min_signal_channel - guard_width,
     max_signal_channel + guard_width]

The current implementation intentionally uses a rectangular context rather than
row-by-row path-following. Rectangular reads are substantially friendlier to
HDF5 and filterbank-style storage because each time range and frequency range is
contiguous. For no-drift and modest-drift narrowband injections, this is both
scientifically reasonable and efficient.

Very large drift spans can make the rectangular context much wider than the
actual signal neighborhood at any one time. A future path-aware mode may read
one rectangle per time chunk and apply a vectorized row-wise mask around the
path, but that is not the default policy today.

Sigma clipping is a fast, consistent first-pass estimator, not a guarantee that
the context is clean. It works well for sparse narrowband outliers and bright
pixels. It can be biased by broad contamination, strong bandpass curvature, or
an asymmetric local environment. Inspect ``NoiseStats.context_bounds``,
``NoiseStats.excluded_bounds``, and ``NoiseStats.n_samples`` when reporting
target-SNR injections.

Current limitations
-------------------

- File-backed synthetic noise mutation is not implemented. Adding synthetic
  noise generally requires broad full-observation mutation.
- ``Frame.data`` on a file-backed frame materializes the full backing file for
  compatibility. Prefer ``read_frame()`` for explicit bounded reads.
- Copy-backed mutation currently performs a full on-disk copy up front. This is
  memory-bounded but still pays full-file disk I/O.
- Noise estimation currently uses one rectangular local context. Path-aware
  chunked noise contexts and richer contamination diagnostics are planned, but
  are not implemented yet.

File-backed spectra and time series
-----------------------------------

The common ``mean`` and ``sum`` reductions are chunked for file-backed frames:

.. code-block:: python

    with stg.Frame.open("observation.h5", mode="r") as frame:
        spectrum = frame.spectrum(mode="mean",
                                  f_range=(6095.0e6, 6095.1e6),
                                  t_range=(0, 300))
        time_series = frame.timeseries(mode="sum",
                                       f_index_range=(1000, 2000),
                                       t_index_range=(0, 128))

These products are returned as normal eager ``Spectrum`` and ``TimeSeries``
objects. Their metadata records that they were produced by integration, which
axis was collapsed, which reducer was used, and which source region was read.
Future reducers such as median or percentile-like products may need different
memory policies.

Best practices
--------------

- Use ``Frame.open_copy()`` for injection into real observations.
- Use ``Frame.open(..., mode="r")`` for inspection and plotting.
- Use ``read_frame()`` to create an eager working region.
- Use ``spectrum()`` and ``timeseries()`` directly for chunked file-backed
  ``mean`` and ``sum`` products.
- Keep signal bounding explicit for custom profiles.
- Estimate SNR from a local window around the injection, with a guard region.
- Treat ``Frame.data`` as a deliberate full-materialization request when the
  frame is file-backed.
