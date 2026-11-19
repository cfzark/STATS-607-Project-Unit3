"""Run the reproducible simulation pipeline from the repository root."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
import csv
from functools import partial
import importlib.metadata
import itertools
import json
import math
from multiprocessing import get_context
from pathlib import Path
import platform
from tempfile import TemporaryDirectory
import time

import numpy as np
from threadpoolctl import threadpool_limits

from .dgps import generate, sparsity
from .methods import fit_lasso, penalty
from .metrics import recovery

REGIME_IDS = {"linear": 0, "sublinear": 1, "fractional": 2}


def load_config(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def condition_parameters(config, regime, p, theta):
    k = sparsity(p, regime, config["sparsity_fraction"], config["sparsity_power"])
    scale = 2 * k * math.log(p - k)
    n = math.ceil(theta * scale)
    alpha = penalty(n, p, k, config["sigma"])
    return k, n, n / scale, alpha


def trial(config, regime, p, theta, replicate, *, parameters=None, copy_X=True):
    if parameters is None:
        parameters = condition_parameters(config, regime, p, theta)
    k, n, theta_realized, alpha = parameters
    key = (REGIME_IDS[regime], p, round(theta * 1_000_000), replicate)
    rng = np.random.default_rng(np.random.SeedSequence(config["seed"], spawn_key=key))
    x, y, beta = generate(n, p, k, rng, config["sigma"], config["beta_min"])
    estimate, diagnostics = fit_lasso(
        x, y, alpha, config["solver_tol"], config["max_iter"], copy_X=copy_X,
    )
    return dict(regime=regime, p=p, k=k, n=n, theta=theta,
                theta_realized=theta_realized, replicate=replicate, **{"lambda": alpha},
                **recovery(beta, estimate, config["zero_tol"]), **diagnostics)


def condition_trials(config, condition, *, reuse_parameters, copy_X):
    regime, p, theta = condition
    parameters = condition_parameters(config, regime, p, theta) if reuse_parameters else None
    return [
        trial(config, regime, p, theta, replicate, parameters=parameters, copy_X=copy_X)
        for replicate in range(config["replicates"])
    ]


def simulate(config, output, *, reuse_parameters=True, copy_X=True, workers=1):
    if workers < 1:
        raise ValueError("workers must be positive")
    start = time.perf_counter()
    raw = output / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    conditions = list(itertools.product(config["regimes"], config["dimensions"], config["theta"]))
    print(f"{len(conditions)} conditions x {config['replicates']} replicates; workers={workers}", flush=True)
    run_condition = partial(condition_trials, config, reuse_parameters=reuse_parameters, copy_X=copy_X)
    pool_context = nullcontext()
    if workers > 1:
        pool_context = ProcessPoolExecutor(
            max_workers=workers, mp_context=get_context("spawn"),
            initializer=threadpool_limits, initargs=(1,),
        )
    with TemporaryDirectory(dir=raw) as directory:
        staged = Path(directory) / "replicates.csv"
        with threadpool_limits(limits=1), pool_context as pool, staged.open("w", newline="") as handle:
            batches = map(run_condition, conditions) if pool is None else pool.map(run_condition, conditions)
            writer = None
            for index, ((regime, p, theta), rows) in enumerate(zip(conditions, batches), 1):
                if writer is None:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                writer.writerows(rows)
                successes = sum(row["success"] for row in rows)
                print(f"[{index:3d}/{len(conditions)}] {regime:10s} p={p} theta={theta:.1f}: "
                      f"{successes}/{config['replicates']}", flush=True)
        staged.replace(raw / "replicates.csv")
    metadata = {
        "config": config,
        "workers": workers,
        "fits": len(conditions) * config["replicates"],
        "simulation_seconds": time.perf_counter() - start,
        "python": platform.python_version(),
        "packages": {p: importlib.metadata.version(p) for p in ("numpy", "scikit-learn", "matplotlib", "threadpoolctl")},
    }
    (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Simulation completed in {metadata['simulation_seconds']:.2f} seconds", flush=True)


def clean(output):
    names = ("raw/replicates.csv", "summary.csv", "run.json", "figures/figure1_rescaling.png",
             "figures/figure2_recovery.png", "figures/diagnostic_convergence.png",
             "figures/performance.png", "stability.json")
    for name in names:
        (output / name).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("all", "simulate", "analyze", "figures", "clean"))
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    output = args.output
    if args.command == "clean":
        clean(output)
        return
    if args.command in ("all", "simulate"):
        simulate(load_config(args.config), output, workers=args.workers)
    if args.command in ("all", "analyze"):
        from .analysis import summarize
        summarize(output)
    if args.command in ("all", "figures"):
        from .figures import plot
        plot(output)


if __name__ == "__main__":
    main()
