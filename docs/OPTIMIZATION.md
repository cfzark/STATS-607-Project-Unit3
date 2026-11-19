# Simulation optimization

## Serial optimization

Unit 2 recalculated `k`, `n`, the realized rescaled sample size, and `lambda` in each call to `trial()`. The serial loop now computes them once per condition. This reduces 37,800 parameter calculations to 189 in the full design.

```python
# Unit 2
for replicate in range(config["replicates"]):
    row = trial(config, regime, p, theta, replicate)

# Unit 3
parameters = condition_parameters(config, regime, p, theta)
for replicate in range(config["replicates"]):
    row = trial(config, regime, p, theta, replicate, parameters=parameters)
```

The [baseline profile](BASELINE.md) also showed time spent on array layout conversion. The benchmark tested changing `Lasso(..., copy_X=True)` to `Lasso(..., copy_X=False)` on top of parameter reuse. This flag affects the fit-side copy and leaves the data generator's layout conversion in place.

### Benchmark

`make benchmark` runs both the serial and parallel comparisons. The serial comparison (`make benchmark-serial`) includes raw CSV output and uses one BLAS thread. Each measurement uses the same 540 trials across 27 conditions. All three versions are warmed up, then measured in three rounds with a rotated run order. [Raw timings](../results/benchmark/serial.csv)

| Version | Median | Range | Difference from baseline median |
|---|---:|---:|---:|
| Baseline | 6.020 s | 5.933–6.826 s | — |
| Reuse condition parameters | 6.120 s | 5.921–6.538 s | +1.7% |
| Reuse parameters and set `copy_X=False` | 6.187 s | 6.049–6.242 s | +2.8% |

The timing differences are small relative to the variation across rounds. The simulation keeps the parameter reuse because it removes work from the inner loop and uses `copy_X=True` by default because the no-copy setting showed no clear gain on this workload.

### Correctness

All 540 paired trials agreed on exact support recovery, false positives, false negatives, sign errors, convergence warnings, and iteration counts. The largest absolute difference in `dual_gap` was `3.87e-15`. Rechecking the final default against the original saved Unit 2 results on the same 540 trials gave the same agreement. All seven Unit 2 tests passed.

## Parallel simulation

The baseline profile attributes nearly all trial time to data generation and Lasso fitting. Each condition can run independently, so `make parallel WORKERS=4` uses a process pool to run several conditions at once. Each process limits BLAS to one thread. Random streams still use condition and replicate identifiers, and the parent writes results in the original order.

```python
# Serial
batches = map(run_condition, conditions)

# Parallel
batches = pool.map(run_condition, conditions)
```

The process pool adds startup time, moves result batches between processes, and can hold several design matrices in memory simultaneously. The benchmark includes process startup and raw CSV output.

### Worker scaling

`make benchmark-parallel` times all 189 conditions with 20 replicates each (3,780 fits). Each worker count ran three times in rotated order. One worker uses the direct serial loop. [Raw timings](../results/benchmark/parallel.csv)

| Workers | Median | Range | Speedup vs. serial |
|---:|---:|---:|---:|
| 1 | 45.159 s | 43.711–46.646 s | 1.00× |
| 2 | 28.107 s | 26.597–29.607 s | 1.61× |
| 4 | 20.608 s | 19.990–22.698 s | 2.19× |

Four workers reduced the median runtime by 54% on this benchmark and are the default for `make parallel`. The benchmark measures scaling at 20 replicates per condition.

### Correctness

All 3,780 paired results agreed at each worker count on recovery, selection errors, solver iterations, convergence warnings, and row order. The largest absolute `dual_gap` difference was `3.87e-15`. `tests/test_regression.py` checks the same agreement on a smaller configuration.

### Formal run

The four-worker run completed all 37,800 fits in **183.59 seconds**, compared with **418.56 seconds** in the recorded Unit 2 run (2.28× as fast). The worker-scaling benchmark above measures the same-session comparison. [Performance figure](../results/figures/performance.png)

The full comparison used the original Unit 2 raw CSV and the same configuration. Every trial and all 189 condition summaries matched on recovery and selection metrics.

To repeat the full validation, generate the original Unit 2 results with `make all` in its checkout, run `make parallel` here, then run `python -m scripts.compare_unit2 ../STATS-607-Project2/results`. The regression test also compares baseline, optimized serial, and optimized parallel execution using only the Unit 3 checkout.

`make stability-check` verifies the saved trials. In the formal run it found no missing or duplicate trials, nonfinite values, convergence warnings, or iteration-limit hits. The largest iteration count was 60, and the captured run log contained no warnings or errors. [Stability report](../results/stability.json)

## Lessons learned

Parallelizing whole conditions gave the largest runtime gain because each worker can generate and fit its own datasets. Hoisting condition parameters removed repeated work, but its effect was small next to generating and fitting 37,800 dense designs. Trying `copy_X=False` helped distinguish the fit-side copy from the array layout conversion in the data generator.
