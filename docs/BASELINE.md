# Unit 2 performance baseline

The baseline is the submitted Unit 2 commit `34ad47c`. Its 189 conditions × 200 replicates took **418.56 seconds** (about 7 minutes) for 37,800 fits. This is the recorded simulation time, excluding summary and figure generation. The baseline uses Python 3.11.7, NumPy 2.2.6, scikit-learn 1.6.0, and one BLAS thread. [Original run metadata](../results/baseline/unit2-run.json)

## Runtime profile

Run `make profile` to profile the original `trial()` with `cProfile`. The run used all three sparsity regimes and dimensions, three values of θ (0.6, 1.2, 2.4), and 10 replicates per condition. The 270 calls took 3.46 seconds under the profiler.

| Component | Cumulative time | Share of trial time |
|---|---:|---:|
| Data generation (`generate`) | 1.72 s | 50% |
| Lasso fitting (`fit_lasso`) | 1.65 s | 48% |

Array layout conversion calls (`asfortranarray`) accounted for 0.78 s within the profiled code. This time overlaps the components above. Data generation and fitting are the main costs, and matrix copying is a candidate for optimization. [Profiler output](../results/baseline/profile.txt)

## Complexity

Run `make complexity` to vary sample size with linear sparsity, p = 512 and k = 205. At six sample sizes, time 24 calls to `trial()` after one warm-up call per size. The median time rose from **14.0 ms** at n = 940 to **106.9 ms** at n = 5,636. The log-log slope was **1.15** over this range.

Generating and moving the dense design takes O(np) time. A full coordinate descent sweep costs up to O(np), so T sweeps cost up to O(Tnp). Median solver iterations fell from 28 to 11 across the grid. The fitted slope therefore describes the observed cost of the whole trial over this grid. [Timing CSV](../results/baseline/complexity.csv) · [Log-log plot](../results/baseline/complexity.png)

## Numerical warnings

The original run recorded **zero convergence warnings across all 189 conditions**.
