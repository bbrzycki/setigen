.. |setigen| replace:: :mod:`setigen`

Frame products and metadata
===========================

Most |setigen| workflows start with a two-dimensional :class:`setigen.Frame`:
time on one axis, frequency on the other, and intensity values in the array.
Common analysis operations collapse, slice, or transform that frame. The goal
is for those derived products to preserve enough context that they remain
scientifically interpretable.

Coordinate conventions
----------------------

Internally, :class:`setigen.Frame` stores data as ``(time, frequency)`` with
frequency channels in increasing order. The ``ascending`` flag controls how the
frame should be written back to filterbank-style products, but frame-level
calculations use increasing frequencies.

Useful frame coordinates include:

``frame.fs`` or ``frame.frequency_centers``
    Frequency-channel center coordinates in Hz.

``frame.frequency_edges``
    Frequency-channel edges in Hz, useful for plotting and region semantics.

``frame.ts`` or ``frame.time_starts``
    Time-bin start coordinates in seconds relative to ``frame.t_start``.

``frame.time_centers``
    Time-bin centers.

``frame.time_edges`` or ``frame.ts_ext``
    Time-bin edges.

When choosing between center and edge conventions, use edges for image extents
and integrated durations. Use centers for evaluating a signal path at the
representative time of each row.

Metadata and headers
--------------------

``frame.metadata`` is the user-facing metadata dictionary. It starts with the
core frame parameters and can be extended with run-specific information:

.. code-block:: python

    frame.add_metadata({
        "drift_rate": 1.5,
        "injection_id": "candidate-0001",
    })

``frame.header`` stores filterbank/HDF5-style observational metadata when a
frame comes from a file or voltage reduction. Save paths rebuild critical header
fields from the frame object, but derived products should still keep headers in
sync when they change dimensions, start time, or frequency bounds.

The current code preserves source name, start time, headers, and custom
metadata through slicing, dedrifting, region reads, and integration. Derived
products also attach a ``metadata["derived"]`` payload that records:

- source shape
- source time/frequency bounds
- operation name
- collapsed axis, when applicable
- reducer, such as ``mean`` or ``sum``
- normalization policy

For example:

.. code-block:: python

    spectrum = frame.spectrum(mode="sum",
                              f_index_range=(100, 200),
                              t_index_range=(0, 16))
    print(spectrum.metadata["derived"])

The exact payload depends on the operation, but integration products include
``operation``, ``product_type``, ``collapsed_axis``, ``reducer``,
``source_shape``, and ``source_bounds``.

Spectrum products
-----------------

A :class:`setigen.Spectrum` is a one-row frame with shape ``(1, fchans)``.
It represents a frame collapsed over time:

.. code-block:: python

    spectrum = stg.spectrum(frame, mode="mean")
    summed_spectrum = frame.integrate(axis="t", mode="sum", as_frame=True)

For spectra:

- ``df`` remains the source channel width.
- ``dt`` is the collapsed time span represented by the product.
- ``fch1`` follows the selected frequency span.
- ``t_start`` is the start of the selected time span.

Use ``mode="mean"`` when you want average power per time bin. Use
``mode="sum"`` when the accumulated power over the integration is the desired
quantity.

Time-series products
--------------------

A :class:`setigen.TimeSeries` is a one-channel frame with shape
``(tchans, 1)``. It represents a frame collapsed over frequency:

.. code-block:: python

    ts = stg.timeseries(frame, mode="mean")
    summed_ts = frame.integrate(axis="f", mode="sum", as_frame=True)

For time series:

- ``dt`` remains the source time resolution.
- ``df`` is the collapsed frequency bandwidth.
- ``fch1`` is the center frequency of the collapsed band.
- ``t_start`` is the start of the selected time span.

Use ``mode="mean"`` for average power over the band. Use ``mode="sum"`` when
the total integrated band power is desired.

Normalization
-------------

Several existing APIs expose normalization options:

- ``integrate(normalize=True)`` sigma-normalizes the reduced product.
- ``Spectrum.plot(snr=True)`` plots a sigma-normalized copy.
- ``TimeSeries.plot(norm=True)`` divides by the mean before plotting.

These remain available for compatibility. For new workflows, prefer naming the
normalization explicitly in metadata and analysis notes. A future shared
collapse spec should distinguish at least:

- no normalization
- sigma-clipped zero-mean/unit-variance normalization
- mean normalization

File-backed products
--------------------

``spectrum()`` and ``timeseries()`` support chunked ``mean`` and ``sum``
reductions for file-backed frames:

.. code-block:: python

    with stg.Frame.open("observation.h5", mode="r") as frame:
        spectrum = frame.spectrum(f_range=(6095.0e6, 6095.1e6),
                                  t_range=(0, 300),
                                  mode="mean")
        ts = frame.timeseries(f_index_range=(100, 500),
                              mode="sum")

The returned products are eager singleton-axis frames. For reducers beyond
``mean`` and ``sum``, first read a bounded region with ``read_frame()`` unless
the reducer explicitly documents chunked file-backed behavior.

Best practices
--------------

- Keep singleton-axis ``Spectrum`` and ``TimeSeries`` products as frame-like
  objects so plotting, units, and serialization remain consistent.
- Use ``mode="mean"`` or ``mode="sum"`` deliberately and record that choice.
- Preserve source context when constructing derived frames manually:
  ``t_start``, ``source_name``, ``header``, and relevant custom metadata.
- For large files, reduce explicit regions rather than accidental full-frame
  materializations.
- Treat derived metadata as part of the scientific record, not only bookkeeping.
