# Array Backend Acceleration Spec

This spec defines one NumPy/CuPy policy for `setigen` so voltage synthesis,
RAW reduction, eager `Frame` work, and file-backed spectrogram injection do not
grow incompatible acceleration APIs.

The existing voltage implementation is the starting point. It already has:

- `ArrayBackend = Literal["auto", "numpy", "cupy"]`
- `stg.voltage.set_backend(...)` and `stg.voltage.get_backend()`
- lazy CuPy import and optional dependency behavior
- legacy `SETIGEN_ENABLE_GPU=1` support
- spec-level `backend` fields in `RawReductionSpec` and
  `VoltageSpectrogramSpec`
- an internal late-bound `xp` proxy for voltage internals

The goal is to lift those conventions to a package-wide array backend layer,
not to create a separate GPU API for `Frame` or file-backed mutation.

## Vocabulary

Use `array backend` for numerical compute backends:

- `numpy`: CPU NumPy arrays.
- `cupy`: GPU CuPy arrays.
- `auto`: resolve through the package default and legacy environment policy.

Use `storage backend`, `file backend`, or `spectrogram backend` for HDF5,
filterbank, GUPPI RAW, and other file-backed I/O. Avoid using a bare
implementation variable named `backend` when both meanings are in scope.

Public APIs may keep a keyword named `backend` where the meaning is already
array compute, especially in voltage specs and CLIs. New code that also handles
storage should use `array_backend` internally or inside config objects.

## Public API Policy

Keep the existing voltage API working:

```python
stg.voltage.set_backend("cupy")
stg.voltage.get_backend()
```

Add a package-level equivalent so non-voltage users do not have to configure a
voltage namespace to accelerate `Frame` rendering:

```python
stg.set_backend("cupy")
stg.get_backend()
```

`stg.voltage.set_backend()` and `stg.set_backend()` should delegate to the same
global state. There must not be separate voltage and frame backend states.

The package-level public surface should stay small:

- `stg.set_backend(backend: Literal["auto", "numpy", "cupy"]) -> None`
- `stg.get_backend() -> Literal["numpy", "cupy"]`

Internal modules can use `setigen._array_backend` for:

- `ArrayBackend`
- `get_array_module(backend=None)`
- `to_numpy(array, xp=None)`
- an internal `xp` proxy when late binding is appropriate

Do not expose a public `stg.xp` proxy initially. It encourages hidden global
behavior in user code and makes callables harder to reason about.

## Resolution Semantics

Resolution must be identical across voltage, frame, and file-backed paths:

- `backend="numpy"` always uses NumPy.
- `backend="cupy"` requires CuPy and raises `ImportError` if CuPy is missing.
- `backend="auto"` follows the package default.
- The package default starts as `"auto"` for compatibility.
- Default `"auto"` follows `SETIGEN_ENABLE_GPU=1`; if CuPy is unavailable in
  this legacy auto mode, it falls back to NumPy.
- Explicit `"cupy"` never falls back silently.
- Operation-level `backend=` overrides the package default for that operation.

`get_backend()` should return the active concrete backend, either `"numpy"` or
`"cupy"`, matching the current voltage behavior.

Do not overload `auto` with performance heuristics. If we later decide that
tiny chunks should stay on CPU to avoid transfer overhead, that should be a
separate policy knob such as `gpu_min_elements`, not a changed meaning of
`backend="auto"`.

## Array Ownership And Transfers

Public data products remain host NumPy arrays unless an API explicitly says
otherwise.

This means:

- `Frame.data` is NumPy.
- `Frame.read_frame(...).data` is NumPy.
- `VoltageSpectrogramResult.data` is NumPy.
- file-backed `read_region()` and `write_region()` exchange NumPy arrays.
- plotting receives NumPy arrays.

Internal compute kernels may use CuPy arrays. The transfer boundary is explicit:

1. read or create a NumPy chunk
2. transfer to the selected array backend with `xp.asarray(...)`
3. render, add, channelize, or reduce on that backend
4. transfer back with `to_numpy(...)` before file writes, plotting, or public
   `Frame` materialization

The current voltage helper `asnumpy()` should become `to_numpy(array, xp=None)`.
It should prefer an explicit `xp` argument when available and otherwise inspect
the array/module safely. This avoids converting with the current global backend
when an array was produced by a different backend.

## Frame And Signal Rendering Policy

`Frame.add_signal()` should eventually accept the same array backend vocabulary:

```python
frame.add_signal(..., backend="auto")
```

For classes or config objects where `backend` would be ambiguous, use
`array_backend`:

```python
SignalRenderConfig(array_backend="auto", compute_dtype=None)
```

Implementation should split signal rendering from mutation before adding GPU
support:

- A renderer builds the signal contribution for an eager array or bounded file
  chunk.
- Eager frames add the rendered contribution to `Frame.data`.
- File-backed frames read a bounded NumPy chunk, render into that chunk, and
  write it back immediately.
- Both paths use the same renderer and therefore the same signal model.

The renderer should accept an array module, not import CuPy directly:

```python
xp = get_array_module(backend)
signal = render_signal_chunk(..., xp=xp, compute_dtype=...)
```

Built-in paths and profiles should be backend-aware where practical. They
should use the provided array module instead of hard-coded `numpy` calls.

Arbitrary user callables in GPU mode are a contract: they must either be
backend-compatible or the user must request `backend="numpy"`. We should raise a
clear error if a callable fails under CuPy rather than silently materializing
large host arrays.

## File-Backed Injection Policy

File-backed GPU acceleration should preserve the current memory goal. It should
never require loading the full observation into memory.

For a writable file-backed frame, each affected chunk follows this pattern:

1. compute the bounded time/frequency region affected by the signal
2. read only that region as a NumPy array
3. convert the chunk to the selected array backend
4. render/add the signal contribution on that backend
5. convert the patched chunk back to NumPy
6. cast according to the storage dtype policy
7. write the region and flush at the existing mutation boundary

The GPU path saves memory in the same way as the CPU file-backed path: memory is
bounded by chunk shape, not observation shape. GPU acceleration changes where
the chunk math happens; it does not change the I/O contract.

The first implementation should optimize correctness and API consistency over
transfer cleverness. Later optimizations can reuse device buffers across chunks
or tune chunk sizes, but those should be invisible to the public API.

## Noise And SNR Policy

`NoiseEstimationConfig` currently uses sigma clipping, which matches the
existing frame behavior and the science preference for local windows around the
injection. Keep that CPU-first for the initial backend work because Astropy
sigma clipping and file-backed local reads already operate on modest host
arrays.

GPU signal rendering can still use CPU-estimated noise stats:

```python
stats = frame.estimate_noise_stats(path=path, config=config)
frame.add_signal(..., snr=snr, noise_stats=stats, backend="cupy")
```

Future GPU noise estimation is allowed, but it should keep the same
`NoiseStats` output and should not change SNR semantics.

## Dtype Policy

Default behavior should preserve current NumPy results as closely as practical.
Do not silently downcast scientific products just because the GPU path exists.

Add an explicit compute dtype later:

```python
SignalRenderConfig(compute_dtype=None)
```

where `None` means preserve current behavior. File-backed writes still cast to
the storage dtype after rendering. If the storage dtype is integer or otherwise
lossy, the write path should make that conversion explicit and tested.

## Randomness Policy

Backend-aware random generation should use the selected module's random
implementation where appropriate. Exact bitwise equivalence between NumPy and
CuPy should not be promised. Tests should compare shape, dtype, deterministic
behavior within a backend, and statistical properties where relevant.

For signal injection, deterministic paths/profiles should remain exactly
comparable across CPU and GPU within reasonable floating-point tolerance.

## Implementation Plan

1. Create `setigen._array_backend` by moving the voltage helper to package
   scope.
2. Leave `setigen.voltage._array_backend` as a compatibility shim that reexports
   the package helper.
3. Reexport `set_backend()` and `get_backend()` from `setigen.__init__`.
4. Update voltage imports to use the package helper or compatibility shim, but
   ensure there is only one global backend state.
5. Add package-level tests for backend validation, legacy environment fallback,
   explicit CuPy errors, and `to_numpy(...)`.
6. Split `Frame.add_signal()` rendering from mutation.
7. Add `backend="auto"` to eager frame signal rendering.
8. Add the same backend resolution to file-backed chunk mutation.
9. Port built-in signal helper functions to backend-aware array operations.
10. Add optional CuPy tests that skip cleanly when CuPy is unavailable.
11. Update docs so voltage, frame, and file-backed injection examples describe
    the same backend policy.

## Validation Requirements

- Existing voltage backend tests continue to pass.
- `stg.set_backend(...)` and `stg.voltage.set_backend(...)` affect the same
  state.
- Explicit `backend="cupy"` raises a clear `ImportError` when CuPy is missing.
- Legacy `SETIGEN_ENABLE_GPU=1` keeps the current auto fallback behavior.
- CPU eager `Frame.add_signal()` remains numerically compatible with current
  behavior.
- CPU and GPU signal rendering match within floating-point tolerance on small
  fixtures when CuPy is installed.
- File-backed CPU and GPU injection produce equivalent patched regions on small
  `.fil` and `.h5` fixtures.
- Plotting after file-backed injection still shows the patched data because
  plotting reads host chunks from the backing file.

## Open Decisions

- Should `Frame.add_signal(..., backend=...)` be added directly, or should
  signal rendering get a config object first?
- Should GPU user-callable failures always raise, or should we offer an
  explicit `callable_backend="numpy"` escape hatch?
- What default chunk size is best when CPU/GPU transfer overhead is included?
- Do we need a public `stg.get_array_module()` for advanced users, or should the
  array module remain internal until a clear use case appears?
- Should `compute_dtype` default to current float behavior forever, or should
  large GPU/file-backed workflows get an opt-in `float32` preset?
