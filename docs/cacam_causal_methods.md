# CACAM Causal Method Exploration

This copy adds a configurable `causal_method` option to CACAM so the same
model can compare several channel dependency matrices:

- `correlation`: absolute Pearson correlation.
- `partial_correlation`: absolute partial correlation from the precision matrix.
- `granger`: Granger causality strength using the best p-value across lags.
- `pcmci`: optional Tigramite PCMCI, falling back to correlation if unavailable.
- `identity`: self-channel only baseline.
- `uniform`: no discovered structure baseline.

Run:

```bash
conda run -n cacam bash scripts/CACAM_causal_methods.sh CalIt2.csv 10
conda run -n cacam bash scripts/CACAM_causal_methods.sh synthetic_glo0.048.csv 3
```

Latest summary:

| Dataset | Method | AUC ROC | Affiliation F |
|---|---:|---:|---:|
| CalIt2 | identity | 0.753608 | 0.650538 |
| CalIt2 | partial_correlation | 0.753254 | 0.651481 |
| CalIt2 | correlation | 0.753249 | 0.651481 |
| CalIt2 | uniform | 0.752950 | 0.650744 |
| CalIt2 | granger | 0.752309 | 0.650330 |
| synthetic_glo0.048 | identity | 0.995882 | 0.850721 |
| synthetic_glo0.048 | correlation | 0.995214 | 0.848149 |
| synthetic_glo0.048 | partial_correlation | 0.994931 | 0.846975 |
| synthetic_glo0.048 | uniform | 0.992351 | 0.842188 |
| synthetic_glo0.048 | granger | 0.991708 | 0.856546 |

Interpretation:

The tested causal discovery variants do not show a consistent improvement over
the simpler baselines. `identity` is the strongest by AUC ROC on both tested
datasets, while `granger` only wins affiliation F on `synthetic_glo0.048` and is
substantially slower. This suggests that the current CACAM architecture is not
benefiting from heavier causal discovery inside each forward pass, at least in
these quick single-seed runs.

Next experimental step:

Run the same comparison on more datasets and seeds before drawing a final paper
claim. If a causal method is kept, prefer precomputing the graph once per
training series instead of recomputing it for every batch forward pass.
