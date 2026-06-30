# study_v2_plain_lvgp — standard (homoscedastic) LVGP, the noise-unaware baseline

Sixth arm of the heteroscedastic mixed-variable Branin BO comparison. Same problem, DOE, sweep
grid and saved `.mat` schema as `study_v2`, but the surrogate is the **plain LVGP** in
`BO_standard_LVGP/` (Yichi Zhang's standard Latent-Variable GP) — one global noise nugget, **no
heteroscedastic / aleatoric model**.

## What's different (the point of this study)

- **Noise-unaware model.** `neg_log_l.m` fits a single homoscedastic σ² + nugget; the model never
  sees per-point noise.
- **Mean only.** Each location is still evaluated with `n_rep` noisy replicates, but only their
  **mean** is passed to `LVGP_fit`. The replicate **variance is recorded** (for the analysis plots
  / σ² comparisons) but is **never** given to the model or the acquisition.
- **ei / lcb / pi only.** The hetero-aware acquisitions (haei/anpei/rahbo) need an aleatoric `r(x)`
  this study deliberately omits, so they are excluded. Sweep = 3 acquisitions × n_rep{3,5,10} ×
  30 seeds = **270 runs**.

`study_driver.m` rebuilds the exact `study_v2` problem/DOE (Branin + heteroscedastic noise, maximin
1/6-inset LHS shared across the 5 categories) and runs the BO loop calling the **unmodified**
`BO_standard_LVGP/*.m` functions; it saves the same field schema (`meta.model="plain_lvgp"`), so
`utils/results.py` and the top-level comparison notebooks load it directly.

## Run it (MATLAB, like study_v2)

```bash
PY=/data/zhq7531/envs/ml_gp_env/bin/python        # for the launcher (scipy/pandas)

# smoke / toy probe: ei,lcb,pi × n_rep{3,5,10} × 1 seed, 2 workers
$PY run_sweep.py --toy --seeds 1 --workers 2

# full sweep: 3 acqs × n_rep{3,5,10} × 30 seeds = 270 runs
$PY run_sweep.py --seeds 30 --workers 12 2>&1 | tee sweep.log
$PY run_sweep.py --collect-only                   # -> sweep_results.csv
```

The launcher starts one shared Xvfb and runs one isolated `matlab -batch study_driver(...)` per
cell (isolated PREFDIR/TMPDIR, retry on transient 5001 license errors) — identical machinery to
`study_v2/run_sweep.py`. Outputs `results/<acf>/nrep<NN>/seed<NN>.mat`.
