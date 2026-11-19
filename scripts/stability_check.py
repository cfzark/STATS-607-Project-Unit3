"""Check saved trials for completeness, finite values, and convergence."""
import argparse
from collections import Counter
import csv
import itertools
import json
from math import isfinite
from pathlib import Path
import re


def check(output, log=None):
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    config = run["config"]
    expected = set(itertools.product(
        config["regimes"], config["dimensions"], config["theta"], range(config["replicates"]),
    ))
    trial_counts = Counter()
    warning_counts = Counter()
    nonfinite_rows = 0
    iteration_limit_hits = 0
    max_iterations = 0
    max_gap = 0.0
    with (output / "raw" / "replicates.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            condition = (row["regime"], int(row["p"]), float(row["theta"]))
            trial_counts[(*condition, int(row["replicate"]))] += 1
            if not all(isfinite(float(value)) for field, value in row.items() if field != "regime"):
                nonfinite_rows += 1
            if row["convergence_warning"] == "1":
                warning_counts[condition] += 1
            iterations = float(row["iterations"])
            if isfinite(iterations):
                iteration_limit_hits += iterations >= config["max_iter"]
                max_iterations = max(max_iterations, int(iterations))
            gap = float(row["dual_gap"])
            if isfinite(gap):
                max_gap = max(max_gap, abs(gap))

    log_issues = []
    if log is not None:
        for line in log.read_text(encoding="utf-8-sig", errors="replace").splitlines():
            if re.search(r"(?:Warning|Error|Exception):|^Traceback |^make.*\*\*\*", line):
                log_issues.append(line.strip())

    checked = sum(trial_counts.values())
    report = dict(
        workers=run.get("workers", 1),
        expected_trials=len(expected),
        checked_trials=checked,
        metadata_fit_count_matches=checked == run["fits"],
        missing_trials=len(expected - trial_counts.keys()),
        extra_trials=len(trial_counts.keys() - expected),
        duplicate_rows=checked - len(trial_counts),
        nonfinite_rows=nonfinite_rows,
        convergence_warnings=sum(warning_counts.values()),
        warning_conditions=[
            dict(regime=regime, p=p, theta=theta, count=count)
            for (regime, p, theta), count in sorted(warning_counts.items())
        ],
        iteration_limit_hits=iteration_limit_hits,
        max_iterations=max_iterations,
        max_abs_dual_gap=max_gap,
        log_checked=log is not None,
        log_issues=log_issues,
    )
    report["passed"] = report["metadata_fit_count_matches"] and not any(
        report[field] for field in (
            "missing_trials", "extra_trials", "duplicate_rows", "nonfinite_rows",
            "convergence_warnings", "iteration_limit_hits", "log_issues",
        )
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--log", type=Path, help="optional UTF-8 console log to scan for warnings and errors")
    args = parser.parse_args()
    report = check(args.output, args.log)
    serialized = json.dumps(report, indent=2, allow_nan=False)
    destination = args.output / "stability.json"
    destination.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    print(f"Saved {destination}")
    if not report["passed"]:
        raise SystemExit("Stability check found issues.")


if __name__ == "__main__":
    main()
