# Contributing

This repository is maintained as a scientific Python library. Changes should
optimize for scientific correctness, reproducibility, public API stability, and
readability.

## Development Setup

Install the package in editable mode with the development extras:

```bash
python -m pip install -e ".[dev]"
```

If you only need documentation dependencies:

```bash
python -m pip install -e ".[docs]"
```

## Running Tests

Run the full test suite:

```bash
pytest -q
```

Run a focused subset while iterating:

```bash
pytest tests/test_frame_injection.py -q
pytest tests/test_voltage/test_raw_voltages.py -q
```

## Building Documentation

Build the Sphinx site locally:

```bash
sphinx-build -b html docs/source docs/build/html
```

## Change Expectations

- Keep changes focused. Avoid unrelated cleanup in the same patch.
- Preserve existing public behavior unless the change is intentional and
  documented.
- Add or update tests for behavioral changes.
- Update README or Sphinx docs when user-facing behavior changes.
- Prefer explicit exceptions over `assert` in library validation.
- Avoid mutable default arguments.
- Add type hints in touched public code where practical.

## Refactoring Guidance

- Prefer extracting cohesive subsystems over introducing one-hop hidden helpers.
- Keep simple public methods inline when extraction only adds indirection.
- Split large modules when they begin mixing concerns such as I/O, plotting,
  validation, and numerical logic.
- Treat private packages such as `setigen._frame` and
  `setigen.voltage._backend` as internal implementation details; they are not
  part of the public compatibility contract.
