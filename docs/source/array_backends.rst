.. |setigen| replace:: :mod:`setigen`

Array backend policy
====================

|setigen| uses NumPy arrays as the default numerical representation. The
``setigen.voltage`` module also supports optional CuPy acceleration for
array-heavy voltage synthesis and reduction work. The backend policy is being
standardized so future frame and file-backed signal rendering can use the same
vocabulary.

Vocabulary
----------

Use ``array backend`` for numerical compute backends:

``numpy``
    CPU NumPy arrays.

``cupy``
    GPU CuPy arrays.

``auto``
    Resolve through the configured default and the legacy environment policy.

Use ``file backend`` or ``storage backend`` for HDF5, filterbank, GUPPI RAW, or
other I/O layers. Keeping these names separate matters because a workflow may
read from an HDF5 storage backend while rendering chunks with a NumPy or CuPy
array backend.

Current public API
------------------

CuPy support is currently public in ``setigen.voltage``:

.. code-block:: python

    import setigen as stg

    stg.voltage.set_backend("cupy")
    print(stg.voltage.get_backend())

Use ``"numpy"`` to force CPU execution:

.. code-block:: python

    stg.voltage.set_backend("numpy")

Use ``"auto"`` to return to the default resolution policy:

.. code-block:: python

    stg.voltage.set_backend("auto")

The legacy environment variable is still supported:

.. code-block:: bash

    export SETIGEN_ENABLE_GPU=1

Resolution semantics
--------------------

The intended semantics are:

- ``backend="numpy"`` always uses NumPy.
- ``backend="cupy"`` requires CuPy and raises an ``ImportError`` if CuPy is not
  installed.
- ``backend="auto"`` follows the configured default.
- The legacy ``SETIGEN_ENABLE_GPU=1`` path may fall back to NumPy when CuPy is
  unavailable.
- Explicit ``"cupy"`` should not silently fall back to CPU.

This behavior already exists for voltage helpers. Future package-wide frame
acceleration should reuse the same meanings rather than introducing separate
GPU flags.

Host arrays at public boundaries
--------------------------------

Public data products should remain host NumPy arrays unless a specific API says
otherwise:

- ``Frame.data`` is NumPy.
- ``Frame.read_frame(...).data`` is NumPy.
- ``VoltageSpectrogramResult.data`` is NumPy.
- File-backed reads and writes exchange NumPy arrays.
- Plotting receives NumPy arrays.

Internal compute kernels may use CuPy arrays, but file I/O, plotting, and
public frame materialization should convert back to host arrays explicitly.

Future file-backed GPU rendering
--------------------------------

The planned frame/file-backed GPU path should be chunk-based:

1. Read a bounded NumPy chunk from the file.
2. Transfer that chunk to the selected array backend.
3. Render and add the synthetic signal contribution.
4. Transfer the patched chunk back to NumPy.
5. Write the chunk to the backing file.

This preserves the large-file memory model. GPU acceleration changes where the
math happens; it should not require loading the full observation.

Best practices
--------------

- Configure the array backend before constructing voltage objects.
- Treat CuPy as optional. Do not make it a required dependency for general
  frame work.
- Keep public arrays on the host unless the API clearly documents otherwise.
- Avoid hidden global backend assumptions in user callables. If a callable must
  run on the GPU, make sure it is compatible with CuPy array operations.
- Do not use ``auto`` as a performance heuristic. If tiny chunks should stay on
  CPU in the future, that should be a separate explicit policy.
