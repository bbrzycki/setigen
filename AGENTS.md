# AGENTS.md

This file defines repository-wide expectations for automated agents and other contributors working in `setigen`.

## Priorities

Changes should optimize for:

- scientific correctness
- public API stability
- reproducibility
- readability
- modular design

## Scope

Treat this repository as a maintained Python package.

- keep changes focused on the package, tests, and documentation
- avoid unrelated cleanup in the same change
- preserve existing user-facing behavior unless a change is intentional and documented

## Required Reading

Before changing voltage synthesis, signal injection, SNR logic, de-drifting workflows, or frequency-placement examples, read [SCIENCE.md](./SCIENCE.md).

Agents should not guess the radio SETI or signal-processing context from code alone when that file applies.

## Scientific Invariants

Do not change these behaviors casually:

- public interfaces that accept or return `astropy.units`-aware values
- ascending and descending frequency ordering semantics
- noise and signal-generation behavior used to interpret SNR, drift, and injected power
- reproducibility controls such as explicit random seeds and generator usage

Behavioral changes to these areas should include tests and documentation updates.

## Module Design

Favor small, readable modules with a single clear responsibility.

- prefer extending an existing focused module over adding more logic to a large general-purpose file
- split code by responsibility when a file starts mixing concerns such as I/O, validation, numerical logic, plotting, and orchestration
- keep public APIs thin and push implementation details into private helpers or submodules
- avoid introducing new "god objects" or large utility files with unrelated behavior
- when touching an already-large file, prefer extracting cohesive pieces into smaller modules rather than continuing to grow it

If a file becomes difficult to understand in one pass, consider splitting it before adding more behavior.

## Editing Rules

- use explicit exceptions for library validation instead of `assert`
- avoid mutable default arguments
- annotate every runtime package function, method, property, and class signature with parameter and return types
- add type hints in touched public code where practical
- use Google-style docstrings for new or materially edited docstrings, with `Args`, `Returns`, and `Raises` sections when they apply
- apply Google-style docstrings consistently to non-trivial classes, functions, and methods; do not use bulk regex rewrites to transform docstrings
- prefer incremental refactors over broad rewrites
- do not mass-reformat unrelated files
- keep comments concise and focused on non-obvious reasoning

## Tests

- add or update tests for every intentional behavior change
- prefer focused tests near the touched functionality before broad suite runs
- keep deterministic tests deterministic

## Documentation

- update README and Sphinx docs when changing user-visible behavior
- update [`SCIENCE.md`](./SCIENCE.md) and related explanatory docs when changing scientific assumptions, core signal-processing behavior, or supported observing configurations
- keep installation and workflow docs aligned with actual project metadata
- do not regenerate or rewrite notebook outputs unless the task explicitly requires it

## Packaging

- keep runtime dependencies separate from test and documentation dependencies
- prefer modern Python packaging metadata in `pyproject.toml`
- declare supported Python versions explicitly

## When In Doubt

- choose the smaller, more reversible change
- document assumptions in code, tests, or commit/PR notes
