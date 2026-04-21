# SCIENCE.md

This document is the scientific context guide for contributors and automated agents working on `setigen`.

It is not a full radio astronomy textbook. It exists to prevent incorrect assumptions when modifying signal synthesis, voltage generation, reduction, or SNR-related workflows.

## When This File Matters

Read this file before changing:

- `setigen.frame` signal injection logic
- `setigen.voltage` signal generation, channelization, quantization, or reduction
- any notebook or example that claims a target SNR, drift rate, or spectral placement
- code that reasons about coarse channels, fine channels, band edges, or de-drifting

## Scientific Goal

`setigen` is used to synthesize and inject radio technosignatures for SETI research.

The dominant use case is narrowband radio signals that appear in time-frequency data as drifting lines. The drift is usually interpreted as relative acceleration between transmitter and telescope. Search pipelines often operate on spectrograms and then integrate over candidate drift rates with de-Doppler methods.

Two different simulation layers matter here:

- `spectrogram` mode: directly synthesizes Stokes I time-frequency data
- `voltage` mode: synthesizes real antenna voltages, passes them through a software signal chain, and then reduces to spectrogram-like products

The voltage path is slower but scientifically closer to telescope backends because it includes quantization, a polyphase filterbank (PFB), and fine channelization.

## Supported Observing Configurations

Many examples in the repo use the simplest voltage case: one antenna, one backend, and one reduced spectrogram product.

That is not the full scientific scope of the voltage layer.

- `setigen.voltage` supports single-antenna workflows
- `setigen.voltage` also supports basic multi-antenna array workflows via `MultiAntennaArray`
- array support includes per-antenna integer delays and shared background-noise modeling across antennas

Agents should not simplify the codebase to “single antenna only” when reasoning about voltage functionality.

## Core Signal Model

In the spectrogram layer, a narrowband signal is described by four factors:

- `path`: central frequency as a function of time
- `t_profile`: intensity as a function of time
- `f_profile`: spectral shape around the central frequency
- `bp_profile`: bandpass weighting as a function of absolute frequency

This is the right mental model for `Frame.add_signal(...)` and `add_constant_signal(...)`. Do not collapse this into “just draw a line”; these factors encode the actual scientific abstraction.

## What the Voltage Module Actually Models

The voltage module models a simplified single-dish signal chain:

1. real voltage streams at the antenna / polarization level
2. optional digitization
3. coarse channelization by a PFB
4. requantization of complex coarse-channel voltages
5. recording to GUPPI RAW
6. later fine channelization and integration into spectrogram products

Important constraint:

- `setigen.voltage` does **not** currently model heterodyne mixing or analog bandpass filtering directly

Instead, `DataStream` and `Antenna` define a synthetic observable band using:

- `sample_rate`
- `fch1`
- `ascending`

So when you inject a voltage signal at frequency `f_start`, that frequency is interpreted relative to this synthetic band definition. Do not assume a hidden RF-to-IF conversion stage exists in code.

## Coarse Channels, Fine Channels, and the “Nice” Part of the PFB

The PFB divides the Nyquist band into coarse channels. Fine channelization then applies a second FFT inside each coarse channel.

This has several consequences:

- coarse-channel edges are not the cleanest place to inject signals
- the middle of a coarse channel is usually the flattest and easiest region for controlled tests
- if you want clean SNR demonstrations, place signals in the flat interior of one coarse channel, not near the transition between coarse channels
- if you want two signals to be visually comparable, keep them in the same coarse channel unless the point of the test is inter-channel behavior

This is the main lesson in `07_efficient_gen_by_subsampling.ipynb`: preserve the signal-processing geometry, then inject where the channel response is well behaved.

## SNR in Voltage Workflows

`setigen.voltage.get_level(...)` is not a generic “make this exact final SNR” function under all conditions.

Its assumptions are narrower:

- unit-variance real Gaussian noise at the voltage stage
- single-polarization amplitude reasoning
- non-drifting signal
- signal centered on a fine spectral bin

So this is the baseline workflow:

1. compute base amplitude with `get_level(...)`
2. multiply by `stream.get_total_noise_std()`
3. correct for off-bin placement with `get_leakage_factor(...)`
4. if drift is large, reason about drift smearing using `get_unit_drift_rate(...)`

Do not claim a notebook demonstrates a target SNR unless the setup accounts for these assumptions or explicitly describes the approximations.

## Spectral Leakage

If a signal is not centered on a fine-channel bin, power leaks into neighboring bins.

In the current `setigen` model:

- power attenuation follows the usual sinc-squared behavior for fine-channel mismatch
- `get_leakage_factor(...)` returns an amplitude correction, not a power correction

That distinction matters because the spectrogram power is proportional to voltage amplitude squared.

When placing synthetic tones for controlled tests:

- prefer exact fine-bin centers when you want a clean SNR calibration
- use deliberate off-bin placement only when testing leakage or realism

## Drift Rates and De-drifting

The unit drift rate is:

- one fine-frequency bin per one time bin in the final spectrogram

In `setigen`, `get_unit_drift_rate(...)` provides this quantity for a chosen backend, `fftlength`, and integration factor.

Interpretation:

- below the unit drift rate, drift effects are present but not trivially approximated
- above the unit drift rate, power is smeared across multiple bins
- de-drifting re-aligns spectra under an assumed drift rate so line-like signals become easier to integrate and score

So for narrowband demonstrations:

- inject with a known drift rate
- reduce to a spectrogram
- use `setigen.dedrift(...)` before quoting final spectral prominence

Do not estimate signal strength from a broad mean spectrum when the signal is drifting or when coarse-channel structure dominates the local baseline.

## Subsampling and Computational Efficiency

High-sample-rate voltage synthesis is expensive because most of the work happens before data are compressed into spectrogram form.

The subsampling notebook demonstrates a scientifically valid shortcut:

- reduce `sample_rate` by a factor `k`
- reduce the number of PFB branches by the same factor `k`

When done carefully, this preserves the basic structure of the reduced product while shrinking the data volume substantially.

This changes practical constraints:

- the maximum possible `num_chans` is reduced
- `block_size` and `num_blocks` interactions may change how much data is recorded

Use this when the scientific question is local spectral behavior, not full-band realism.

## Efficiency Rules That Follow from the Science

Because the voltage path is expensive, efficient workflows are part of scientific correctness:

- record only the coarse channels you need
- if you only care about one narrowband test, do not synthesize a wide band and crop later
- prefer reduction to `.fil` / `.h5` on disk over keeping large intermediate products in memory
- use subsampling when it preserves the signal-processing behavior relevant to the experiment
- use explicit memory budgets for large fine FFT examples

The library should prefer bounded working sets over optimistic large allocations.

## Interpreting Frequency Coordinates Correctly

Do not guess sign conventions.

Always check:

- `ascending`
- `fch1`
- `chan_bw`
- whether a frequency is an absolute observing frequency or a local offset from a coarse-channel center

For voltage examples that aim to place a signal in the clean middle of one coarse channel, compute:

- the coarse-channel center from backend metadata
- the fine-bin offset from `fftlength`
- the absolute injection frequency from those two pieces

## What Counts as a Good Controlled Example

A good notebook or test for voltage injection should usually do all of the following:

- place signals in the flat interior of a chosen coarse channel
- use `get_level(...)` and `get_leakage_factor(...)` rather than hand-picked amplitudes
- state the assumptions behind any claimed SNR
- use a narrow frequency crop around the injected signal
- use de-drifting before comparing drifting-signal prominence
- keep runtime and memory bounded

## What Not to Guess

Do not guess:

- that a requested SNR in the notebook equals the exact measured final SNR
- that frequency placement is correct unless it is derived from backend geometry
- that a signal near a coarse-channel edge behaves like one in the center
- that raw-voltage examples should synthesize the full available bandwidth by default
- that the voltage module includes analog mixing / IF selection

## Canonical References

Primary paper:

- Bryan Brzycki et al. 2022, *Setigen: Simulating Radio Technosignatures for the Search for Extraterrestrial Intelligence*, *The Astronomical Journal*, 163:222
- DOI: `10.3847/1538-3881/ac5e3d`
- Accepted manuscript: <https://par.nsf.gov/servlets/purl/10585711>

Repository examples that capture important practical usage:

- [05_raw_file_gen_snr.ipynb](./jupyter-notebooks/voltage/05_raw_file_gen_snr.ipynb)
- [07_efficient_gen_by_subsampling.ipynb](../setigen_development/stg-79_voltage_centered_snr_drift/07_efficient_gen_by_subsampling.ipynb)

## Keep In Lockstep

If a change modifies core functionality, scientific assumptions, supported observing configurations, signal-processing semantics, or the interpretation of SNR / drift / frequency placement, update this file and any linked explanatory documentation in the same change.

Agents should treat stale scientific documentation as a correctness issue, not a polish issue.

## For Agents

Before editing voltage or SNR-sensitive code, summarize these four things explicitly:

1. what physical quantity is being controlled: voltage amplitude, power, SNR, drift rate, or frequency placement
2. what stage of the pipeline the change affects: antenna voltages, coarse channelization, fine channelization, or spectrogram analysis
3. whether the signal is being placed in a clean part of the PFB response
4. what the memory and file-I/O consequences are for the proposed workflow

If you cannot answer those four questions, you do not understand the change well enough yet.
