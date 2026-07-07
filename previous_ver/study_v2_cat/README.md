# study_v2_cat — joint categorical-GP Bayesian optimization (Method C)

Third arm of the heteroscedastic mixed-variable BO study. Same problem and same sweep grid
(12 acquisition configs × n_rep{3,5,10} × 30 seeds) as the other two; the difference is
purely **how the 5 categories are modeled**:

| study | model of the 5 categories | shares across categories? |
|---|---|---|
| `study_v2` (LVGP, MATLAB) | one latent-variable GP | yes — learned latent space |
| `study_v2_gp` (per-category) | 5 independent botorch GPs | **no** |
| `study_v2_cat` (here, **Method C**) | one `MixedSingleTaskGP` + `CategoricalKernel` | yes — categorical kernel |

So this is the **botorch analogue of LVGP's sharing**: one joint GP over `(x1, category)`
with a `CategoricalKernel` on the category dimension and the replicate variance as fixed
heteroscedastic noise (the "Method C" prototype from `heterockedastic_new_inv/heterosk.ipynb`,
cells 44–52). Three-way head-to-head: `compare_studies_multi([...])`.

## What differs from study_v2_gp
**Only the GP.** Everything else — `problem.py`, `acquisitions.py`, the **aleatoric noise
model**, the BO loop (`bo.py`), `run_sweep.py`, `results.py`, plotting — is identical, so
the comparison isolates exactly one variable: 5 independent GPs vs 1 joint categorical GP.
- `utils/model.py` — `MixedCategoryGP`: one `MixedSingleTaskGP(cat_dims=[1], train_Yvar=…)`
  instead of 5 `SingleTaskGP`s. Same `fit / levels / predict / mean_std` interface, so
  `utils/bo.py` is reused verbatim (only the model import alias + the `meta['model']` tag change).

The categorical choice is still handled by **enumeration** in the BO loop (optimize each
category's acquisition over `x1`, take the best) — the joint GP just provides the posterior
at each `(x1, category)`.

## Run it
```bash
PY=/data/zhq7531/envs/ml_gp_env/bin/python
cd /data/zhq7531/IDEAL/hetero_lvgp/study_v2_cat

$PY run_sweep.py --toy --workers 6 --seeds 3                  # quick probe
$PY run_sweep.py --seeds 30 --num-iter 30 --workers 18        # full 1080-run grid
$PY bo_runner.py --acf haei --param 1.0 --n-rep 10 --seed 1   # single cell
$PY run_sweep.py --collect-only                               # rebuild CSV
$PY save_all_plots.py     # gallery + comparison/ (pairwise + 3-way vs LVGP & per-cat GP)
$PY tb_monitor.py --interval 20   # live progress -> tb/  (TensorBoard on a free port, e.g. 6008)
```

**Runtime note:** the joint `MixedSingleTaskGP` is pricier to fit than 5 tiny independent
GPs, so a run is **~25–40 s** (vs ~15 s for study_v2_gp). Full sweep ≈ **~30 min @ 18
workers** (~18 min @ 28). Resumable; pins BLAS/torch to 1 thread per worker. (The per-run
cost growing with iterations — the joint GP sees all points — is itself a finding worth a
runtime plot.)

## Plotting & the 3-way comparison
Reuses study_v2's full `StudyResults` gallery (loader reads `.npz` and `.mat`), plus:
- `compare_studies(cat, other, …)` — pairwise overlay.
- `compare_studies_multi([(lvgp,"LVGP"), (gp,"Per-category GP"), (cat,"Categorical GP")], …)`
  — the 3-way overlay (one subplot per acquisition, one line per study).

`save_all_plots.py` auto-loads `../study_v2/results` (LVGP) and `../study_v2_gp/results`
(per-category GP) and writes pairwise + 3-way figures to `plots/comparison/`.

## Fidelity / notes
- Same `.npz` field schema as `study_driver.m` (so all plotting is shared).
- `meta['model'] = 'mixed_cat'` tags these runs.
- Category encoded as ordinal code `level-1` in input column 1; `CategoricalKernel`
  (via `cat_dims=[1]`) handles it — **no one-hot**. Only x1 is input-normalized.
- Seeds: numpy RNG ≠ MATLAB RNG, so the DOE matches the *protocol*, averaged over 30 seeds.
