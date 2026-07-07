# study_v2_gp — per-category GP Bayesian optimization (Part 2)

Python counterpart to the MATLAB **LVGP** study (`../study_v2`). Same heteroscedastic
mixed-variable Branin problem, same sweep grid (12 acquisition configs × n_rep{3,5,10} ×
30 seeds), but the **opposite modeling choice**:

| | model of the 5 categories |
|---|---|
| `study_v2` (LVGP) | ONE latent-variable GP — **shares** information across categories |
| `study_v2_gp` (here) | **one independent botorch GP per category** — NO sharing |

LVGP's advantage is exactly that cross-category sharing, so this per-category baseline
isolates how much it buys you. **Headline:** `compare_studies(gp, lvgp, ...)` overlays the
two on identical axes. (A future third arm — botorch *categorical* `MixedSingleTaskGP`,
"Method C" — is stubbed in the notebook, not implemented.)

## Layout
```
run_sweep.py        parallel, resumable grid launcher + CSV collector (ProcessPoolExecutor)
bo_runner.py        run ONE (acf, param, n_rep, seed) cell from the CLI
save_all_plots.py   regenerate the whole plots/ tree (reused study_v2 gallery + comparison/)
utils/
  problem.py        objective f_true, heteroscedastic noise, DOE, level/config metadata
  model.py          per-category GP fit/predict (Method B; notebook fit_botorch_fixed)
  aleatoric.py      per-category degree-2 log-variance poly r(x) for the hetero acqs
  acquisitions.py   ei, lcb, pi, haei, anpei, rahbo  (ported from acquisition_func.m)
  bo.py             the BO loop: per-iter fit → enumerate categories → argmax acquisition
  results.py        study_v2's StudyResults gallery, loader extended to .npz + .mat,
                    plus compare_studies() for the LVGP-vs-GP overlay
results/<acf_tag>/nrep<NN>/seed<NN>.npz     (same field schema as study_driver.m)
plots/{main,analysis,single_runs,comparison}/
notebooks/visualize.ipynb
```

## Run it
Use the **ml_gp_env** interpreter (botorch 0.14), NOT base anaconda (0.16):
```bash
PY=/data/zhq7531/envs/ml_gp_env/bin/python

# quick probe (5-config subset, resumable):     ~13 s/run
$PY run_sweep.py --toy --workers 8 --seeds 5

# full grid: 12 configs × n_rep{3,5,10} × 30 seeds = 1080 runs   (~12 min on 18 workers)
$PY run_sweep.py --seeds 30 --num-iter 30 --workers 18

# single cell (debug):
$PY bo_runner.py --acf haei --param 1.0 --n-rep 10 --seed 1 --num-iter 30

# rebuild CSV only / all plots:
$PY run_sweep.py --collect-only
$PY save_all_plots.py            # main + analysis + single_runs + comparison/
```
`run_sweep.py` is **resumable** (skips cells whose `.npz` exists) and pins BLAS/torch to 1
thread per worker so the outer pool doesn't oversubscribe the cores.

## How the BO loop handles the categorical variable
The category is NOT modeled by a GP — it is handled by **enumeration**. Each iteration:
1. fit one GP + one aleatoric poly **per category** on that category's data;
2. for each category, optimize the acquisition over `x1 ∈ [-5, 10]` (dense grid + L-BFGS-B);
3. take the **best acquisition value across the 5 categories** → next `(category, x1)`;
4. evaluate the noisy objective `n_rep×`, append to that category's data.

All categories' acquisitions share ONE global incumbent (`min` observed sample-mean), so the
values are comparable across categories — this is what makes step 3 valid, and it mirrors
`Heter_BO_GF/find_next.m`.

## Design notes / fidelity to the LVGP study
- **Same problem** as `study_v2/study_driver.m`: Branin, `x1∈[-5,10]`, 5 levels
  `VAR_FCTR=[15,2,8,0,10]`, per-level noise `×[10,7,9,5,12]`, 2 shared maximin-LHS DOE
  points/level, `n_rep` replicates.
- **Acquisitions** ported line-for-line from `acquisition_func.m` (EI not LogEI, `lcb=μ−2s`,
  PI margin 0.01, plus haei/anpei/rahbo with their knobs). Aleatoric `r(x)` mirrors
  `fit_aleatoric_polymodel` (degree-2 ridge log-σ), fit **per category** (no shared latent).
- **Per-category GP** = notebook `fit_botorch_fixed`: `SingleTaskGP` with replicate variance
  as fixed heteroscedastic noise (`Normalize`+`Standardize`).
- **`.npz` schema** matches `study_driver.m` field-for-field, so `StudyResults` plots this
  study and the LVGP study with the same code (its loader reads both `.npz` and `.mat`).
- **Seeds:** numpy RNG ≠ MATLAB RNG, so a seed does NOT reproduce the MATLAB DOE
  byte-for-byte; we replicate the *protocol* and average over 30 seeds (fair, not identical).

## Status
Validated on a 5-config × 5-seed toy run: single-cell + all 6 acquisition families produce
complete/finite output; the parallel pool, resumability, and CSV collection work; all 20
standard plots + the overlays render on the `.npz` data. Toy results show the expected
signature (LVGP ≤ per-category GP in true regret on EI/HAEI/ANPEI). Ready for the full
30-seed sweep.
