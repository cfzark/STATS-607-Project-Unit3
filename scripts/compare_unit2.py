"""Compare a complete Unit 3 run to the original Unit 2 output."""
import argparse
import csv
import json
from pathlib import Path

from .benchmark import compare


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="Unit 2 results directory")
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    reference_run = json.loads((args.reference / "run.json").read_text(encoding="utf-8"))
    current_run = json.loads((args.output / "run.json").read_text(encoding="utf-8"))
    if reference_run["config"] != current_run["config"]:
        raise ValueError("Unit 2 and Unit 3 used different simulation configurations")

    results = {}
    for label, name in (("replicates", "raw/replicates.csv"), ("conditions", "summary.csv")):
        reference = read_rows(args.reference / name)
        current = read_rows(args.output / name)
        exact_rows, gap_difference = compare(reference, current)
        results[label] = dict(
            compared=len(current), exact_rows=exact_rows, max_abs_dual_gap_difference=gap_difference,
        )
        print(f"{label}: {len(current)} match within tolerance; max gap difference {gap_difference:.3g}")
    if results["replicates"]["compared"] != current_run["fits"]:
        raise ValueError("Replicate count differs from run metadata")

    before = reference_run["simulation_seconds"]
    after = current_run["simulation_seconds"]
    report = dict(
        same_config=True, unit2_seconds=before, unit3_seconds=after,
        workers=current_run["workers"], recorded_speedup=before / after, **results,
    )
    destination = args.output / "benchmark" / "full_comparison.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Recorded full-run speedup: {report['recorded_speedup']:.2f}x. Saved {destination}.")


if __name__ == "__main__":
    main()
