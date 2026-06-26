# study_v2_acqf — joint categorical GP + BoTorch `optimize_acqf_mixed`

Fourth arm of the heteroscedastic mixed-variable Branin BO comparison. **Same problem,
DOE, sweep grid and saved schema as the other studies — the only thing that changes is
how the acquisition is optimized.**

| study | model | acquisition + optimizer |
|-------|-------|-------------------------|
| `study_v2` | LVGP (MATLAB), shares across categories | MATLAB closed-form acqs + fmincon |
| `study_v2_gp` | 5 independent per-category GPs | hand-rolled closed-form acqs + grid/L-BFGS |
| `study_v2_cat` | one joint `MixedSingleTaskGP` (shares) | hand-rolled closed-form acqs + grid/L-BFGS |
| **`study_v2_acqf`** | **one joint `MixedSingleTaskGP` (shares)** | **BoTorch `optimize_acqf_mixed`** |

So `study_v2_acqf` vs `study_v2_cat` isolates *the acquisition optimizer* (BoTorch's
native mixed optimizer vs the hand-rolled grid), on an identical model.

## How the acquisition is done (the point of this study)

BoTorch maximizes, our problem minimizes, so the model is fit on **−y** (`utils/model.py`):
posterior mean `= −μ_f`, `best_f = max(−y_obs) = −min(y_obs)`. The acquisition is then
optimized with **`optimize_acqf_mixed`**, which enumerates the 5 categorical codes
(`fixed_features_list`) and gradient-optimizes `x1` within each (`utils/bo.py`).

`utils/acqf.py` — the factory:
- `ei`  → BoTorch `LogExpectedImprovement`
- `lcb` → BoTorch `UpperConfidenceBound(beta=4)`  (`mean + √4·σ = −μ_f + 2σ`, i.e. minimize `μ_f − 2σ`)
- `pi`  → BoTorch `ProbabilityOfImprovement`
- `haei`/`anpei`/`rahbo` → **custom `AnalyticAcquisitionFunction` subclasses** (not in
  BoTorch); they combine the GP posterior with a torch-differentiable aleatoric `r(x)`
  (`TorchAleatoric`, wrapping the per-category degree-2 log-variance polynomials), ported
  from `acquisition_func.m`.

**Note on the optimizer:** the het-aware acquisitions (`haei`, `anpei`) are near-flat in
this heavy-noise problem (`scale = 1 − √r/√(s²+γ²r) ≈ 0` since `r ≫ s²`), so L-BFGS can't
climb them — we use a **dense `raw_samples=256` grid-like initialization** (`NUM_RESTARTS=10`)
so the argmax is actually found. That is the BoTorch analogue of the other studies' grid optimizer.

## Run it

```bash
PY=/data/zhq7531/envs/ml_gp_env/bin/python      # botorch 0.14.0 (NOT base anaconda)

# single cell (debug)
OMP_NUM_THREADS=1 $PY bo_runner.py --acf rahbo --param 1.0 --n-rep 10 --seed 1 --num-iter 30

# toy probe (resumable)
OMP_NUM_THREADS=1 $PY run_sweep.py --toy --workers 8 --seeds 2

# full sweep: 12 configs x nrep{3,5,10} x 30 seeds = 1080 runs (~33 min on 18 workers)
OMP_NUM_THREADS=1 $PY run_sweep.py --workers 18 2>&1 | tee sweep.log
$PY -c "import run_sweep; run_sweep.collect()"   # -> sweep_results.csv
```

Outputs `results/<acf_tag>/nrep<NN>/seed<NN>.npz`, name-for-name with the other studies, so
the `StudyResults` gallery and the top-level `compare_methods.ipynb` overlays load it directly
(`meta.model = "botorch_acqf"`).
