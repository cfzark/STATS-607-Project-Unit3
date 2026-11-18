"""Time and compare three serial versions on the same simulation trials."""
from contextlib import redirect_stdout
import csv
from io import StringIO
import json
from math import isclose
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory

from src.simulation import load_config, simulate


VERSIONS = {
    "baseline": dict(reuse_parameters=False, copy_X=True),
    "reuse_parameters": dict(reuse_parameters=True, copy_X=True),
    "reuse_parameters_no_copy": dict(reuse_parameters=True, copy_X=False),
}


def run(config, output, version):
    with redirect_stdout(StringIO()):
        simulate(config, output, **VERSIONS[version])
    seconds = json.loads((output / "run.json").read_text())["simulation_seconds"]
    with (output / "raw" / "replicates.csv").open(newline="") as handle:
        return seconds, list(csv.DictReader(handle))


def compare(reference, actual):
    if len(reference) != len(actual):
        raise AssertionError(f"Expected {len(reference)} results, got {len(actual)}")
    exact_rows = 0
    max_gap_difference = 0.0
    for index, (expected, row) in enumerate(zip(reference, actual), 1):
        exact_rows += expected == row
        for field, value in expected.items():
            if field == "dual_gap":
                before, after = float(value), float(row[field])
                max_gap_difference = max(max_gap_difference, abs(before - after))
                if isclose(before, after, rel_tol=1e-8, abs_tol=1e-12):
                    continue
            elif row[field] == value:
                continue
            raise AssertionError(f"Row {index}, {field}: {value} != {row[field]}")
    return exact_rows, max_gap_difference


def main():
    config = load_config("config.json")
    config.update(theta=[0.6, 1.2, 2.4], replicates=20)
    versions = list(VERSIONS)
    timings = []
    reference = None
    with TemporaryDirectory() as directory:
        output = Path(directory)
        warmup = dict(config, theta=[1.2], replicates=2)
        for version in versions:
            run(warmup, output, version)
        for offset in range(len(versions)):
            for version in versions[offset:] + versions[:offset]:
                seconds, rows = run(config, output, version)
                if reference is None:
                    reference = rows
                exact_rows, gap_difference = compare(reference, rows)
                timings.append(dict(
                    round=offset + 1, version=version, fits=len(rows), seconds=seconds,
                    exact_rows=exact_rows, max_abs_dual_gap_difference=gap_difference,
                ))
                print(f"Round {offset + 1} | {version}: {seconds:.3f} s, "
                      f"{exact_rows}/{len(rows)} exact rows", flush=True)

    destination = Path("results/benchmark/serial.csv")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(timings[0]))
        writer.writeheader()
        writer.writerows(timings)
    for version in versions:
        seconds = median(row["seconds"] for row in timings if row["version"] == version)
        print(f"Median | {version}: {seconds:.3f} s")
    print(f"All trial comparisons passed. Saved timings to {destination}")


if __name__ == "__main__":
    main()
