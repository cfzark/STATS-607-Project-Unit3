"""Plot observed full-run times and worker scaling."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    benchmark = args.output / "benchmark"
    full = json.loads((benchmark / "full_comparison.json").read_text(encoding="utf-8"))
    timings = defaultdict(list)
    with (benchmark / "parallel.csv").open(newline="", encoding="utf-8") as handle:
        measurements = list(csv.DictReader(handle))
    for row in measurements:
        timings[int(row["workers"])].append(float(row["seconds"]))
    fits, = {int(row["fits"]) for row in measurements}
    workers = sorted(timings)
    speedup = [median(timings[1]) / median(timings[count]) for count in workers]

    fig, (formal, scaling) = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
    bars = formal.bar(
        ["Unit 2 serial", f"Unit 3 ({full['workers']} workers)"],
        [full["unit2_seconds"], full["unit3_seconds"]], color=["#69788a", "#176b75"], width=0.6,
    )
    formal.bar_label(bars, fmt="%.1f s", padding=4)
    formal.margins(y=0.17)
    formal.set(title=f"Recorded full runs ({full['replicates']['compared']:,} fits)", ylabel="Simulation time (s)")

    scaling.plot(workers, speedup, "o-", color="#176b75", linewidth=2)
    for count, factor in zip(workers, speedup):
        scaling.annotate(f"{factor:.2f}×", (count, factor), xytext=(0, 8),
                         textcoords="offset points", ha="center")
    scaling.set(title=f"Worker scaling ({fits:,} fits per run)", xlabel="Workers",
                ylabel="Speedup vs. serial", xticks=workers, ylim=(0.8, max(speedup) + 0.3))
    scaling.margins(x=0.12)
    scaling.grid(axis="y", alpha=0.25)
    for axis in (formal, scaling):
        axis.spines[["top", "right"]].set_visible(False)
    destination = args.output / "figures" / "performance.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180)
    plt.close(fig)
    print(f"Saved {destination}")


if __name__ == "__main__":
    main()
