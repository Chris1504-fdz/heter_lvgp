# study_v2_hgpr — per-category heteroscedastic GP (HGPR + polynomial noise + MAP loss)

Fifth arm of the heteroscedastic mixed-variable Branin BO comparison. Same problem, DOE,
sweep grid and saved schema as the others; the surrogate is **Method A from
`heterockedastic_new_inv/heterosk.ipynb`** — the HGPR of *Ozbayram, Olivier, Graham-Brady
(CMAME 2024)*.

## What's different about the model

One **HGPR per category** (5 independent, like `study_v2_gp`), but each one **learns its own
polynomial noise model jointly with the GP** via a MAP loss:

$$\sigma^2(x) = \bigl(k_{\mathrm{al}}\,e^{\sum_{i=1}^d \theta_i x^i}\bigr)^2,\qquad
\mathrm{Loss} = -\log p(y\mid\ell,k_{\mathrm{al}},\theta) + R_\ell + R_{k_{\mathrm{al}}} + R_\theta$$

with an ARD-RBF kernel and Gaussian/log-normal priors on $(\theta, k_{\mathrm{al}}, \ell)$. Unlike the
other studies (a GP on replicate means + a **separately**-fit aleatoric polynomial), here the
aleatoric `r(x)` used by haei/anpei/rahbo **is** the HGPR's learned $\sigma^2(x)$ — model and noise
are trained together. The HGPR is fit on the **raw replicates** (it learns $\sigma^2$ from the scatter).

| study | per-category model | noise / aleatoric `r(x)` |
|-------|--------------------|--------------------------|
| `study_v2_gp` | `SingleTaskGP`, replicate-var as fixed noise | separate degree-2 log-σ poly |
| **`study_v2_hgpr`** | **HGPR, MAP-trained** | **learned jointly: σ²(x)=(k_al e^{Σθx})²** |

## Hyperparameters (validated — read this)

Set to the config that reproduces the notebook's **Experiment 1 mean MAE = 9.11**:
**`poly_degree=2`** (log-σ is exactly quadratic in x1 here → well specified), **`mu_l=-1.5`,
`lambda_l_sq=1.0`**, `mu_k_al=-1.0`, `lambda_k_al_sq=1.0`, `lambda_theta_sq=1.0`, Adam `lr=0.01`.

⚠️ The tight length-scale prior **`mu_l=-2.5, lambda_l_sq=0.05`** (used for the Part-1 synthetic in
the notebook) forces `l≈0.08`, overfits, and scores **MAE 14.3** — do not use it here.

**Cross-checked against the paper (ref.pdf, Appendix B).** Eqs. 19–21 match the loss terms exactly,
and the paper's *explicit* prior-variance selections are the ones used here: `λ²_θ = 1/p` (B.1,
Gaussian, decreasing with #features), **`λ²_kal = 1`** (B.2, "balanced choice" between 0.1 and 5),
**`λ²_l = 1`** (B.3). The paper warns that `λ²_l = 0.1` is *"tightly concentrated around small
lengthscales ... overly sensitive ... prone to overfitting"* — i.e. it independently predicts the
failure of the old tight-prior config. Normalization to `[0,1]` is also from the paper (here x uses
the fixed domain `[LB,UB]` for warm-start consistency; the paper uses the data range).

`x` is normalized to `[0,1]` over the **fixed** domain `[LB,UB]` (keeps the length-scale prior
meaningful and enables warm-starting); `y` over its data range. See `utils/model.py`.

## Cost control

A cold 2000-epoch fit is ~8 s (50 pts) – 36 s (300 pts). Each BO iteration changes only one
category, so we **cold-fit all 5 once, then warm-refit only the changed category** (300 epochs,
~0.6 s). Expect tens of seconds per run (timing printed by the validation / `meta.runtime`).

## Run it

```bash
PY=/data/zhq7531/envs/ml_gp_env/bin/python      # botorch 0.14.0 / torch (NOT base anaconda)

OMP_NUM_THREADS=1 $PY bo_runner.py --acf rahbo --param 1.0 --n-rep 10 --seed 1 --num-iter 30
OMP_NUM_THREADS=1 $PY run_sweep.py --toy --workers 8 --seeds 2          # resumable probe
OMP_NUM_THREADS=1 $PY run_sweep.py --workers 18 2>&1 | tee sweep.log    # full 1080-run sweep
$PY -c "import run_sweep; run_sweep.collect()"                          # -> sweep_results.csv
```

Outputs `results/<acf_tag>/nrep<NN>/seed<NN>.npz` (`meta.model="hgpr"`), name-for-name with the
other studies, so the `StudyResults` gallery and the top-level comparison overlays load it directly.
