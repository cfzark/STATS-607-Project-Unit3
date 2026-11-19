"""Benchmark condition-level multiprocessing against the serial simulation."""
import csv
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory

from src.simulation import load_config
from .benchmark import compare, run


def main():
    config = load_config("config.json")
    config["replicates"] = 20
    worker_counts = (1, 2, 4)
    timings = []
    reference = None
    with TemporaryDirectory() as directory:
        output = Path(directory)
        for offset in range(len(worker_counts)):
            for workers in worker_counts[offset:] + worker_counts[:offset]:
                print(f"Round {offset + 1} | starting workers={workers}", flush=True)
                seconds, rows = run(config, output, workers=workers)
                if reference is None:
                    reference = rows
                exact_rows, gap_difference = compare(reference, rows)
                timings.append(dict(
                    round=offset + 1, workers=workers, fits=len(rows), seconds=seconds,
                    exact_rows=exact_rows, max_abs_dual_gap_difference=gap_difference,
                ))
                print(f"workers={workers}: {seconds:.3f} s; {len(rows)} rows agree "
                      f"(max dual_gap difference {gap_difference:.3g})", flush=True)

    destination = Path("results/benchmark/parallel.csv")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(timings[0]))
        writer.writeheader()
        writer.writerows(timings)
    medians = {
        workers: median(row["seconds"] for row in timings if row["workers"] == workers)
        for workers in worker_counts
    }
    for workers, seconds in medians.items():
        print(f"Median | workers={workers}: {seconds:.3f} s, {medians[1] / seconds:.2f}x speedup")
    print(f"All paired comparisons passed. Saved timings to {destination}")


if __name__ == "__main__":
    main()
