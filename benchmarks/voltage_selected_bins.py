from __future__ import annotations

import argparse
import os
import time
from collections.abc import Iterable

import numpy as np

import setigen as stg


def _parse_counts(raw_counts: str) -> list[int]:
    """Parse a comma-separated list of selected coarse-channel counts."""
    return [int(item.strip()) for item in raw_counts.split(",") if item.strip()]


def _time_call(repeats: int, func: object) -> float:
    """Return the best runtime across repeated calls."""
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        func()
        timings.append(time.perf_counter() - start)
    return min(timings)


def run_benchmark(
    *,
    num_taps: int,
    num_branches: int,
    windows: int,
    counts: Iterable[int],
    repeats: int,
    seed: int,
) -> None:
    """Benchmark full-FFT slicing against selected-bin PFB channelization."""
    rng = np.random.default_rng(seed)
    samples = rng.standard_normal(num_taps * num_branches * windows)

    print(
        "backend={backend} taps={taps} branches={branches} windows={windows} repeats={repeats}".format(
            backend="cupy" if os.getenv("SETIGEN_ENABLE_GPU") == "1" else "numpy",
            taps=num_taps,
            branches=num_branches,
            windows=windows,
            repeats=repeats,
        )
    )
    print("channels,full_seconds,selected_seconds,speedup")

    for count in counts:
        full_filterbank = stg.voltage.PolyphaseFilterbank(
            num_taps=num_taps,
            num_branches=num_branches,
        )
        selected_filterbank = stg.voltage.PolyphaseFilterbank(
            num_taps=num_taps,
            num_branches=num_branches,
        )

        full_seconds = _time_call(
            repeats,
            lambda: full_filterbank.channelize(
                samples,
                cache=False,
                start_chan=0,
                num_chans=count,
                method="full",
            ),
        )
        selected_seconds = _time_call(
            repeats,
            lambda: selected_filterbank.channelize(
                samples,
                cache=False,
                start_chan=0,
                num_chans=count,
                method="selected",
            ),
        )
        speedup = full_seconds / selected_seconds if selected_seconds else float("inf")
        print(f"{count},{full_seconds:.8f},{selected_seconds:.8f},{speedup:.3f}")


def main() -> None:
    """Run the selected-bin voltage benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-taps", type=int, default=8)
    parser.add_argument("--num-branches", type=int, default=1024)
    parser.add_argument("--windows", type=int, default=16)
    parser.add_argument("--counts", default="1,2,4,8,16,64")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_benchmark(
        num_taps=args.num_taps,
        num_branches=args.num_branches,
        windows=args.windows,
        counts=_parse_counts(args.counts),
        repeats=args.repeats,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
