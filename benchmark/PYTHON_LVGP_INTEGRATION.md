# Python LVGP integration & the aida compute node — working notes

*Personal notes (gitignored). Documents how the Python heteroscedastic LVGP was made to match the
MATLAB engine, the second compute node (aida) it runs on, every file changed and why, and how to
re-run things.*

---

## 1. Goal

The benchmark compares surrogate models for heteroscedastic mixed-variable BO. Two of the four models
(`standard_LVGP`, `heter_LVGP`) run in **MATLAB**, which is slow (~41 s/iteration, ~6–7 h per cell
under load) and license-bound. The objective here was:

1. Get a **Python LVGP** that reproduces the MATLAB LVGP's *behaviour* (validated against the stored
   MATLAB results), so the LVGP cells can be produced ~12–25× faster.
2. Stand up a **second machine (`aida`)** to run those Python cells *additively*, without touching the
   MATLAB sweep already running on `vision`.

Both are done. The Python LVGP is a **faithful numpy port of the MATLAB code** (not a third-party
package) — see §4. It matches MATLAB and does not exhibit the failure the third-party package did.

---

## 1b. Modifications relative to the original lvgp-bayes GitHub

Short answer: we changed **exactly one line-level thing** in the cloned lvgp-bayes repo, then
**stopped using its model** in favour of a from-scratch port of the MATLAB LVGP. Full accounting:

### The ONLY direct edit to the lvgp-bayes clone
`testing_lvgp_bayes/` is a clone of `https://github.com/syerramilli/lvgp-bayes.git`. `git diff` shows
**one changed file** (4 insertions, 1 deletion):

**`lvgp_bayes/optim/__init__.py`** — made the MCMC backend import optional:
```python
 from .mll_scipy import fit_model_scipy
-from .numpyro_hmc import run_hmc_numpyro
+try:
+    from .numpyro_hmc import run_hmc_numpyro   # optional MCMC backend (needs jax+numpyro)
+except ImportError:                            # the scipy MAP path above does not require it
+    run_hmc_numpyro = None
```
Why: the package eagerly imports `numpyro`→`jax`, which (a) isn't installable on aida's old glibc and
(b) we never use — we only call the scipy-MAP path `fit_model_scipy`, not the fully-Bayesian HMC.
That's the entire patch; nothing else upstream was touched.

### Why we don't use the rest of lvgp-bayes (the real "modification")
We first wrapped lvgp-bayes (`utils/models/lvgp_torch.py`), but its modeling choices **diverge from the
MATLAB reference** we validate against, so we wrote our own port
(`utils/models/hetero_lvgp_native.py`) mirroring MATLAB instead:

| aspect | lvgp-bayes (original GitHub) | MATLAB / our native port |
|---|---|---|
| response scaling | standardize y (mean/std) | **min-max to [0,1]** |
| observation noise | `FixedNoiseGaussianLikelihood` on standardized y | replicate variances added to the **correlation matrix** |
| priors | horseshoe prior shrinking noise → tiny | **none** (profiled `sigma2`) |
| conditioning | reactive Cholesky jitter | **eps-ladder eigenvalue-floor nugget** |
| aleatoric r(x) | per-category Python poly (`AleatoricModels`) | **one pooled ridge poly over [x, latent-z]** |
| hyperparameter fit | L-BFGS (`fit_model_scipy`) | **SLSQP + Sobol multistart + eps-continuation** |

These aren't bugs in lvgp-bayes — it's a valid, more-Bayesian implementation. They just make it a
*different* method than the MATLAB LVGP, so it can't be a faithful Python stand-in. The native port
reproduces MATLAB (validated: same optima, latent embedding, within ~1%).

### New files added (none exist in the original GitHub)
- `utils/models/hetero_lvgp_native.py` — the MATLAB-faithful port (DEFAULT LVGP; §4.1, §5b).
- `utils/models/lvgp_torch.py` — the lvgp-bayes wrapper (reference-only now; §4.3).
- `validate_torch.py`, `compare_lvgp.py`, `notebooks/compare_matlab_python.ipynb` — validation + comparison.
- registry/plumbing edits: `utils/models/__init__.py`, `utils/bo.py`, `separate_gp.py`, `categorical_kernel.py` (§4).

---

## 2. The two machines

| | `vision` (primary) | `aida` (secondary) |
|---|---|---|
| role | MATLAB sweep + primary Python env | Python LVGP production/validation |
| cores | 18 physical / 36 logical | 6 physical / 12 logical |
| OS / glibc | modern | **Ubuntu 16.04, glibc 2.23 (EOL)** |
| Python env | `/data/zhq7531/envs/ml_gp_env` (torch 2.7) | `~/mc` (miniconda, torch 2.3.1) |
| SSH | — | `ssh zhq7531@aida.mech.northwestern.edu` |

`aida`'s ancient glibc is the whole reason its setup was fiddly (§3). Docker was **not** an option
there — the account is not in the `docker` group and has no sudo.

---

## 3. The aida environment (how it was built)

aida's glibc 2.23 is too old for a normal modern-Python install. The working recipe (three failed
attempts before this):

1. **Older Miniconda installer** — modern Miniconda requires glibc ≥ 2.28; used
   `Miniconda3-py310_23.3.1-0` (pre-2.28) which installs on 2.23. Lives at `~/mc` on aida.
2. **torch 2.3.1 CPU** — its Linux wheels are manylinux2014 (glibc 2.17), so they *run* on 2.23.
   (torch 2.4+ switched to glibc 2.28 wheels — would NOT work. lvgp-bayes needs torch ≥ 2.3, so 2.3.1
   is the sweet spot.) Call `~/mc/bin/python` / `~/mc/bin/pip` **directly** — `conda activate` silently
   no-ops under `nohup` and drops back to system Python 2.7.
3. **Runtime deps only** — `numpy scipy gpytorch botorch` (skip `pandas`/`scikit-learn`: analysis-only,
   and they try to compile from source on aida's gcc 5.4). aida only *produces* result cells; all
   analysis happens on vision.
4. **lvgp-bayes patched import** — see §4.4 (`optim/__init__.py`).

The benchmark code + the shared DOE cache are rsynced to `~/hetero_bench/` on aida. Analysis is never
run there.

---

## 4. Code changes (files, paths, why)

### 4.1 THE MAIN DELIVERABLE — faithful MATLAB LVGP port (NEW)

**[`benchmark/utils/models/hetero_lvgp_native.py`](utils/models/hetero_lvgp_native.py)** — *new file.*

A from-scratch numpy re-implementation of the MATLAB LVGP, mirroring
`benchmark/matlab/heter_lvgp/LVGP_fit_noise.m`, `neg_log_l_noise.m`, `LVGP_predict_noise.m`,
`corr_mat.m`, `to_latent.m`. One engine serves **both** Python LVGPs via a `HETERO` flag:

- `LVGPNative` (homoscedastic, noise-blind acqs `ei/lcb/pi`) — analogue of `standard_LVGP` / MATLAB
  `LVGP_fit.m`. Noise off; conditioning from the eps-ladder nugget only.
- `HeterLVGPNative` (heteroscedastic, all 6 acqs) — analogue of `heter_LVGP` / MATLAB
  `LVGP_fit_noise.m`. Replicate variances enter the correlation matrix.

What it reproduces exactly (these are the things the third-party package got *differently*, which is
why it failed — see §5):
- **min-max normalization** of X and Y to [0,1] by *training* range (MATLAB), not std-standardization.
- **Gaussian correlation kernel** `exp(−Σ 10^φ·Δ²)`; latent dims use φ=0.
- **latent embedding**: level 1 at origin; dim_z=2 with the 2nd coord of the first free level fixed
  at 0 (identifiability).
- **heteroscedastic noise added to the correlation matrix** as `Σ/(Ymax−Ymin)²`, with `sigma2`
  (process variance) profiled out.
- **eps-ladder eigenvalue nugget** (`10^(-1:-0.5:-8)`, continuation warm-starts, likelihood-selected
  level) guaranteeing `min-eig(R) ≥ eps` before every Cholesky.
- **exact predict/MSE formulas**, including MATLAB's fit-vs-predict subtlety: stored `R` = correlation
  + eigen-nugget (no noise); prediction forms `R_pred = R + diag(noise/scale2)/sigma2` and re-derives
  β and the posterior from it. Epistemic (latent) variance uses unit-diagonal `R_new_new`.

Predictions are returned in **raw Y units**. The aleatoric `r(x)` comes from the shared
`AleatoricModels` (same as every other Python model), so all acquisitions are engine-consistent.

### 4.2 Registry

**[`benchmark/utils/models/__init__.py`](utils/models/__init__.py)** — registered the two native
models as the **default** Python LVGPs: `lvgp_native` (BLIND) and `heter_lvgp_native` (FULL). The
lvgp-bayes wrapper (§4.3) is still registered, but guarded in a `try/except ImportError` and kept only
as an A/B reference — it is **not** the default.

### 4.3 lvgp-bayes wrapper (NEW, now reference-only)

**[`benchmark/utils/models/lvgp_torch.py`](utils/models/lvgp_torch.py)** — *new file.* First attempt:
wraps the third-party `lvgp-bayes` package (`LVGPTorch`, `HeterLVGPTorch`). It works for the
homoscedastic case but the heteroscedastic case gets *stuck* on some seeds (§5), which is why the
native port (§4.1) replaced it as the default. Kept for reference/comparison. Contains the fixes from
the debugging journey (raw-scale de-standardization, `NUGGET` floor) documented in §5 — those were
necessary but not sufficient, hence the port.

### 4.4 lvgp-bayes package patch (for aida / no-jax)

**`testing_lvgp_bayes/lvgp_bayes/optim/__init__.py`** — the package eagerly did
`from .numpyro_hmc import run_hmc_numpyro`, which imports `jax`. We only use the scipy-MAP path
(`fit_model_scipy`), never the fully-Bayesian HMC path, and jax on glibc 2.23 is another rabbit hole.
Wrapped the numpyro import in `try/except ImportError` so the package loads without jax/numpyro.
(`testing_lvgp_bayes/` is itself gitignored; note recorded here for reproducibility.)

### 4.5 BO loop plumbing

**[`benchmark/utils/bo.py`](utils/bo.py)** — the per-iteration fit call now passes the problem bounds:
`model = model_cls.fit(data, needs_r=True, bounds=bounds)`. The LVGP models need the true bounds for
input normalization; the other models ignore the kwarg (see §4.6).

### 4.6 Existing models absorb the new kwarg

**[`benchmark/utils/models/separate_gp.py`](utils/models/separate_gp.py)** and
**[`benchmark/utils/models/categorical_kernel.py`](utils/models/categorical_kernel.py)** — `fit()`
signature gained `**_kw` so they harmlessly ignore `bounds=`/`warm_from=` passed by `bo.py`.
Behaviour-neutral.

### 4.7 Validation harness (NEW)

**[`benchmark/validate_torch.py`](validate_torch.py)** — compares the Python LVGP against the stored
MATLAB results *distributionally* (per acquisition × n_rep, 30 seeds). Uses paired Wilcoxon + unpaired
Mann-Whitney on the best-found-noiseless distributions and prints an EQUIVALENT/DIFFER verdict per
cell, plus median±IQR convergence bands. Two-sided on purpose: a drop-in must be neither worse **nor**
better than MATLAB. Pairs `lvgp_native ↔ standard_LVGP` and `heter_lvgp_native ↔ heter_LVGP`.
Distributional (not cell-by-cell) because the initial design is shared (CRN) but BO iteration noise is
not (`default_rng([seed,1])` in Python vs `rng(seed)` in MATLAB).

### 4.8 Run harness timeout (unrelated to LVGP, done same session)

**[`benchmark/run.py`](run.py)** (lines ~52, ~58) — the MATLAB kill-timeout was raised from 8 h → **16 h**
(d < 9) and 40 h → **72 h** (d ≥ 9), because golinski heter cells stretch to 6–8 h under multi-user
contention and were at risk of being killed mid-run. It's a hang-guard, not a scheduler.

---

## 5. The debugging journey (why the port was necessary)

The validation caught two real problems in the lvgp-bayes wrapper before the port:

1. **Scale bug (fixed in `lvgp_torch.py`).** The wrapper returned predictions in lvgp-bayes's internal
   *standardized-y* space, while the incumbent `ymin` and the acquisitions work in *raw* y. EI/PI and
   the heteroscedastic acqs over-explored; LCB (scale-invariant argmin) masked it. Fix: de-standardize
   `mu`/`var` back to raw units in `predict()`, and standardize the fixed noise. This fixed the medians.

2. **Heteroscedastic sticking (the reason for the port).** A minority of seeds (e.g. heter lcb/anpei/
   rahbo seed02/03) stayed stranded at the initial-design best (~2.8) instead of finding the optimum
   (~0.47). *Not* numerical conditioning — a nugget floor changed nothing. Direct debugging showed the
   BO trapped, repeatedly sampling a **high-noise boundary point** whose epistemic uncertainty never
   collapses under `FixedNoiseGaussianLikelihood`. This is a **heteroscedastic-noise modeling
   difference**: lvgp-bayes (std-standardization + FixedNoise + horseshoe priors) ≠ MATLAB (min-max +
   noise-on-correlation + profiled sigma2). Porting MATLAB's actual treatment (§4.1) resolved it:
   seed02 2.80→0.466, seed03 2.95→0.614, all recommending the correct level, matching MATLAB.

---

## 5b. Fit optimizer — the latent embedding (Path A done; Path B for later)

After the port matched MATLAB on the stuck seeds, a subtler gap remained: on some seeds native's
**latent fit landed at a worse local optimum than MATLAB's**. Fitting native on MATLAB's *exact* data,
its latent had `lv2` pinned at the z-bound (±3) and used the 2nd latent dimension, while MATLAB found a
clean 1-D embedding (`lv2 ≈ 1.63`). Head-to-head NLL: **MATLAB −518.4 vs native −508.6** — native was
**stuck ~10 nll above** MATLAB (under-optimized, not over-optimized). The inflated latent separations
inflated the epistemic variance ~2–3×, which made LCB/EI over-explore and pushed ANPEI onto the
safe-but-suboptimal level 4.

**Root cause:** native used `scipy L-BFGS-B` (projected quasi-Newton), which slid the latent onto the
box edge. MATLAB uses `fmincon('Algorithm','interior-point')`, whose barrier keeps iterates interior.

**Path A — DONE (faithful).** Switched the fit to **SLSQP** over the *free* hyperparameters
(scrambled-Sobol multistart, eps-ladder continuation, finite differences kept — MATLAB's `my_grad` is
forward-differences, so no analytic gradient here). SLSQP reaches the **same** optimum as MATLAB's
interior-point fmincon (verified nll −518.37; latent matches to ~0.01 across seeds) and is ~14× cheaper
than scipy's true interior-point `trust-constr` (which also matched but cost ~48 s/fit vs ~5 s). See the
long comment at the SLSQP block in [`utils/models/hetero_lvgp_native.py`](utils/models/hetero_lvgp_native.py).

Also fixed in the same pass: the **pooled aleatoric poly** (§4.1) — native previously used the shared
per-category Python poly (`AleatoricModels`), which inflated `r` ~4× on under-sampled levels; the port
now fits ONE ridge poly over `[x, latent-z]`, matching MATLAB `fit_aleatoric_polymodel` (r = 1.09 vs
MATLAB 1.10 at the optimum).

**Path B — NOT DONE, a direction to look if the fit ever needs to be faster or more robust (esp. in
higher dimensions).** Give the profile marginal likelihood an **analytic gradient** and pass it to the
optimizer, instead of finite differences. This is *not* what MATLAB does (MATLAB finite-differences via
`my_grad`, step 1e-8), so it's a deliberate departure — treat it as an implementation optimization, not
a fidelity change. Expected upside: ~5–10× faster fits (no finite-difference gradient/Hessian passes)
and more reliable convergence as the hyperparameter count grows (φ per dim + 2·(L−1) latent coords).
**Caveats to watch:** (1) a stronger optimizer may find a *lower-NLL* optimum than MATLAB's
finite-difference fmincon ever reaches — then native would diverge by being *more* optimal, and you'd
have to decide whether to match MATLAB deliberately or accept native as better; (2) the gradient must
be carried through the latent embedding *and* the eigenvalue-floor nugget, which is where the
derivation is fiddly. Where to implement: the `nll_free`/`_profile` path in `hetero_lvgp_native.py`
(return `(value, grad)` and set `jac=True`).

---

## 6. How to run it

**Native port sanity/stuck-seed check (aida):**
```bash
ssh zhq7531@aida.mech.northwestern.edu
cd ~/hetero_bench/benchmark
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
  ~/mc/bin/python test_native.py 50 1,2,3 lcb    # iters, seeds, acqs
```

**Full validation vs MATLAB (aida, ~3.3 h, 540 cells):**
```bash
cd ~/hetero_bench/benchmark
~/mc/bin/python run.py --functions branin_hetero \
  --models lvgp_native heter_lvgp_native --seeds 30 --num-iter 50 --workers 8
```
Then pull results to vision and score:
```bash
# on vision, after rsyncing aida's results/branin_hetero/{lvgp_native,heter_lvgp_native} back
python3 validate_torch.py --torch-root <pulled> --matlab-root results
```

**Sync code vision → aida** (never sync `results/` back the wrong way):
```bash
rsync -az --exclude __pycache__ benchmark/utils/ \
  zhq7531@aida.mech.northwestern.edu:~/hetero_bench/benchmark/utils/
```

---

## 7. Status / open items

- Native port **built, wired in as default, verified** on the stuck seeds (matches MATLAB, no
  outliers, 3.5 s/it).
- Full 540-cell distributional validation **running on aida**; the EQUIVALENT/DIFFER verdict from
  `validate_torch.py` is the acceptance gate before using the port to produce production cells.
- If it passes: aida produces the heteroscedastic LVGP cells additively, in parallel with the vision
  MATLAB sweep — the fast path for the remaining d<9 problems and the 10-D block.
- The vision MATLAB sweep (`tmux bench2`) runs untouched throughout.
