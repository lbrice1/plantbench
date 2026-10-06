# Operating regimes in `reactor_separator_recycle`'s design space

The demonstration study of the library: data mining and knowledge discovery on a library of closed-loop trajectories generated across `reactor_separator_recycle`'s heat-integrated designs. It asks two questions:

1. Do settled, unsettled and limit-cycling behavior separate in a low-dimensional embedding of the late window of each run, and which variables separate the clusters?
2. Does the early window, the first 100 min after the upset, predict the late outcome?

## Reproduction

```
.venv/bin/pip install -e ".[study]"
.venv/bin/plantbench generate studies/regimes/spec.toml --workers 10   # about 1.5 h on 10 cores
.venv/bin/python -m studies.regimes.analysis data/regimes
.venv/bin/python -m studies.regimes.figures
```

`pilot.toml` is the 210-run pilot that fixed the size of the dataset.

## Files

| File | Contents |
|---|---|
| `spec.toml` | the dataset: 7 plants × 6 upsets × 48 Latin-hypercube points over the reactor temperature tuning and the analyzer deadtime, 2016 runs |
| `labels.py` | the outcome of a run, read from its last 300 min |
| `dataset.py` | loads the runs, their design-space coordinates and outcomes |
| `discovery.py` | the steps of the data mining workflow of Briceno-Mena et al. (2022) and Seghers et al. (2023) |
| `analysis.py` | the analysis; writes `results/results_regimes.txt` and `results/analysis.npz` |
| `figures.py` | draws `figures/r1_late_embeddings.png`, `r2_sgs.png`, `r3_cost_and_early.png` |

Every number reported from this study is in `results/results_regimes.txt`.

## Decisions and their order

- The outcome criterion in `labels.py` was written before any run of the study was generated, and it was not changed afterwards.
- The pilot fixed the size of the dataset at 48 points per combination. Limit cycles made up 4% of the pilot's runs, so this size gives about 80 in the full dataset.
- The pipeline follows the data mining workflow of Briceno-Mena et al. (2022) and Seghers et al. (2023): sampling with averaging, z-score normalization, PCA at 95% of the variance, t-SNE and PaCMAP with their defaults, HDBSCAN, the Davies–Bouldin index and subspace greedy search, with every random draw seeded and one change to the search (`discovery.py`).
- Two choices were made on the pilot and are reported as such:
  - The late window is analyzed as deviations from the design point (global structure) and as deviations from each run's own late-window mean (local structure). On the pilot, the global representation clustered by upset rather than by outcome.
  - Variables whose spread is below 1e-8 of their level are dropped before normalization (the cleaning step). In the local representation the reflux, ratioed to a constant flow, varies only by round-off, and z-scoring turned that round-off into a unit-variance variable that dominated both the clusters and SGS.
- The early-window classifier (10 nearest neighbors, 5-fold stratified cross-validation, with its baselines) was fixed before any result on the full dataset was seen.
- The first generation of the dataset integrated each run in one call. One run of 2016 stepped from rest across its upset into a column composition with negative mole fractions, and the solver accepted the resulting NaN as success. The library now restarts the integration at every declared discontinuity and rejects non-finite trajectories, and both the pilot and the dataset were regenerated with it. The outcome mix of the pilot was unchanged (79%, 18%, 4%).
- The comparison of the analyzer deadtime with the tuning of the temperature loop (section 1 of the results file) was added after the first full analysis, to support a statement in the paper; the analysis is seeded and every other number was reproduced unchanged.

## References

- Briceno-Mena, L. A., Nnadili, M., Benton, M. G. and Romagnoli, J. A. (2022). Data mining and knowledge discovery in chemical processes: effect of alternative processing techniques. *Data-Centric Engineering*, 3, e18.
- Seghers, E. E., Briceno-Mena, L. A. and Romagnoli, J. A. (2023). Unsupervised learning: local and global structure preservation in industrial data. *Computers & Chemical Engineering*, 178, 108378.
