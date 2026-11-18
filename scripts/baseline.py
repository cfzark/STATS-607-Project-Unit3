"""Measure the original Unit 2 trial with one BLAS thread."""
import argparse
import cProfile
import csv
import itertools
from pathlib import Path
import pstats
from statistics import median
from time import perf_counter
import warnings

import numpy as np
from threadpoolctl import threadpool_limits

from src.simulation import load_config, trial


def profile(config, output):
    conditions = list(itertools.product(config["regimes"], config["dimensions"], (0.6, 1.2, 2.4)))
    replicates = 10
    rows = []
    profiler = cProfile.Profile()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with profiler:
            for regime, p, theta in conditions:
                for replicate in range(replicates):
                    rows.append(trial(config, regime, p, theta, replicate))

    with (output / "profile.txt").open("w", encoding="utf-8") as report:
        print(f"{len(conditions)} conditions x {replicates} replicates; one BLAS thread", file=report)
        print(f"Convergence warnings: {sum(r['convergence_warning'] for r in rows)}", file=report)
        print(f"Other warnings emitted to caller: {len(caught)}", file=report)
        print(f"Maximum solver iterations: {max(r['iterations'] for r in rows)}", file=report)
        for row in rows:
            if row["convergence_warning"]:
                print(f"regime={row['regime']}, p={row['p']}, theta={row['theta']}, "
                      f"replicate={row['replicate']}", file=report)
        for warning in caught:
            print(warnings.formatwarning(
                warning.message, warning.category, warning.filename, warning.lineno
            ).strip(), file=report)
        pstats.Stats(profiler, stream=report).strip_dirs().sort_stats("cumulative").print_stats(25)


def complexity(config, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = max(config["dimensions"])
    rows = []
    for theta in (0.4, 0.8, 1.2, 1.6, 2.0, 2.4):
        trial(config, "linear", p, theta, 0)
        durations = []
        iterations = []
        for replicate in range(1, 25):
            start = perf_counter()
            row = trial(config, "linear", p, theta, replicate)
            durations.append(perf_counter() - start)
            iterations.append(row["iterations"])
        rows.append(dict(p=p, k=row["k"], n=row["n"], theta=theta, replicates=len(durations),
                         median_ms=1000 * median(durations), median_iterations=median(iterations)))

    with (output / "complexity.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    n = np.array([row["n"] for row in rows])
    ms = np.array([row["median_ms"] for row in rows])
    slope, intercept = np.polyfit(np.log(n), np.log(ms), 1)
    fig, ax = plt.subplots(figsize=(6.5, 4), layout="constrained")
    ax.loglog(n, ms, "o", label="Median of 24 trials")
    ax.loglog(n, np.exp(intercept) * n**slope, "--", label=f"Log-log slope: {slope:.2f}")
    ax.set(
        xlabel="Sample size n",
        ylabel="Time per trial (ms)",
        title=f"Unit 2 baseline: linear sparsity, p={p}, k={rows[0]['k']}",
    )
    ax.grid(alpha=0.2, which="both")
    ax.legend()
    fig.savefig(output / "complexity.png", dpi=160)
    plt.close(fig)
    print(f"Log-log slope: {slope:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("profile", "complexity"))
    args = parser.parse_args()
    config = load_config("config.json")
    output = Path("results/baseline")
    output.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        command = profile if args.command == "profile" else complexity
        command(config, output)
    print(f"Saved {args.command} results to {output}")


if __name__ == "__main__":
    main()
