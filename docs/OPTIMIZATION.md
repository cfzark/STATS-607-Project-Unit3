# Serial optimization

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

## Benchmark

`make benchmark` times the serial simulation, including raw CSV output, with one BLAS thread. Each measurement uses the same 540 trials across 27 conditions. All three versions are warmed up, then measured in three rounds with a rotated run order. [Raw timings](../results/benchmark/serial.csv)

| Version | Median | Range | Difference from baseline median |
|---|---:|---:|---:|
| Baseline | 6.020 s | 5.933–6.826 s | — |
| Reuse condition parameters | 6.120 s | 5.921–6.538 s | +1.7% |
| Reuse parameters and set `copy_X=False` | 6.187 s | 6.049–6.242 s | +2.8% |

The timing differences are small relative to the variation across rounds. The simulation keeps the parameter reuse because it removes work from the inner loop and uses `copy_X=True` by default because the no-copy setting showed no clear gain on this workload.

## Correctness

All 540 paired trials agreed on exact support recovery, false positives, false negatives, sign errors, convergence warnings, and iteration counts. The largest absolute difference in `dual_gap` was `3.87e-15`. Rechecking the final default against the original saved Unit 2 results on the same 540 trials gave the same agreement. All seven Unit 2 tests passed.
