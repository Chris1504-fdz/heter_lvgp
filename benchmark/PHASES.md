# Roadmap — remaining phases

Living checklist. **Do these one at a time**, in order. Update the boxes as they land.

Equations source: `resources/Mixed_Variable_Heteroscedastic_Test_Problems_Eng_Problems_v2.pdf`
Budgets source:   `resources/init_doe_iter.xlsx` (rounded → `doe.PROBLEM_GRID`)

---

## Status

| | problems | state |
|---|---|---|
| ✅ done | branin_hetero, sixhump_camel, griewank_2d, ackley_2d | 1 continuous + 1 categorical; swept, analysed |
| ❌ todo | griewank_10d (9-D), ackley_10d (9-D), rastrigin_6d (5-D), golinski (6-D), piston (6-D), otl_circuit (5-D) | **stubs — equations not written, and the framework is 1-D only** |

**6 of 10 problems remain** (not 5 of 9).

---

## ✅ PHASE 2a — generalize the framework to *d* continuous dimensions — COMPLETE

The blocker. The equations are *not* the hard part: every layer currently assumes exactly
**1 continuous dim + 1 categorical** (`X = [x1, level]`, 2 columns).

**Acceptance test (non-negotiable): the 4 existing 1-D problems must reproduce the CURRENT results
BIT-EXACTLY.** The generalized code must reduce to the present code when d = 1. That is the whole
validation strategy — we already have trusted results to regress against.

To keep bit-exactness at d = 1, three things must reduce exactly:
1. `make_doe_nd` must not consume extra RNG at d=1 (skip the per-dim shuffle) → identical DOE + noise.
2. The d-dim acquisition optimizer must reduce to the current `linspace(256)` + L-BFGS-from-top-3 at d=1.
3. The d-dim aleatoric polynomial must reduce to `[1, w, w²]` at d=1.

### Regression oracle
`regress_1d.py` re-runs a spread of stored 1-D cells and diffs bit-for-bit. **Baseline confirmed: current
pipeline reproduces its own results to worst |Δ| = 0.00e+00** (6 python cells), so the oracle is real.
Run `python regress_1d.py` after every step (add `--matlab` for the 2 LVGP cells).

### Checklist
- [x] `utils/doe.py` — `make_doe_nd`: skip the dim-shuffle when d==1. **Verified: d=1 identical points +
      identical rng state; d>1 SLHD invariants hold.**
- [x] `utils/problems.py` — `ProblemSpec`: vector `bounds` + `.d`; `f(X,level)`/`sigma(X,level)` take
      (n,d) for d>1, bare (n,) x1 for d==1 (existing 1-D f untouched). **Regression bit-exact (6/6).**
      _(still TODO in this file: `initial_doe` → `make_doe_nd` — that's Step 3.)_
- [x] `utils/problems.py::initial_doe` → `make_doe_nd`; `X_sample` (n, d+1). **Bit-exact vs every
      sampled stored doe_cache file**; d>1 smoke (13 pts = 4+4+5 remainder rule) OK. `doe_cache` needed
      no change (stores whatever width initial_doe emits).
- [x] `utils/bo.py` — per-level data holds `X` (n,d); `_minimize_box`: **d==1 = the verbatim legacy
      grid+polish (bit-exact); d>1 = deterministic unscrambled-Sobol (2^⌈log2 max(256,128d)⌉ cands) +
      L-BFGS-B polish from top max(3,d)**. Deliberately NOT botorch optimize_acqf: the acquisitions
      (incl. haei/anpei/rahbo + the aleatoric poly) are closed-form numpy; wrapping them in torch would
      fork the acquisition definitions between 1-D and d-dim. Same architecture, scaled.
- [x] `utils/models/base.py` — aleatoric poly d-dim: per-column powers [1, W, W²], **NO cross terms —
      verified against BOTH MATLAB builders** (build_poly_features + _local are pure per-col powers).
- [x] `utils/models/{separate_gp,categorical_kernel}.py` — (n, d) inputs; cat_dims=[d],
      Normalize(d+1, indices=0..d-1). **Regression bit-exact 6/6 + 3-D end-to-end smoke PASS**
      (both models find the true optimum of a synthetic 3-D×3-level problem; rahbo works).
- [x] `matlab/{heter,standard}_driver.m` — `nd = numel(lb)`, `ind_qual = nd+1`, objective over
      `X(:,1:nd)`, `x_next(1:end-1)` / `x_next(end)`; v2 true-f columns generalized. Vendored
      find_next was already d-general (`d = size(X_sample,2)`, row-vector lb/ub).
- [x] `utils/results.py` — loader keeps stored width (n, d+1); level = LAST column everywhere.
      All 5040 stored runs still load; tables unchanged.
- [x] **REGRESSION: full `regress_1d.py --matlab` = PASS 8 | FAIL 0 at tol=0** — all 6 python cells
      AND both MATLAB cells (heter_LVGP + standard_LVGP, real 50-iter runs) reproduce the stored
      results bit-for-bit through the complete d-dim refactor. d>1 additionally validated by a 3-D
      python end-to-end smoke + a run-twice determinism proof (|Δ|=0). d>1 MATLAB smoke lands with
      the first real multi-dim equation (Phase 2c vehicle = rastrigin_6d).

---

## ▶ PHASE 2b — cost probe  (BEFORE committing to any multi-dim sweep)  ◀ CURRENT

A crude O(n³) projection from the *measured* 475 s/cell (branin, heter_LVGP, 50 iters, n_tr 10→60):

| problem | n_tr grows | iters | projected heter_LVGP |
|---|---|---|---|
| rastrigin_6d | 32 → 232 | 200 | **~29 h/cell** |
| golinski | 30 → 230 | 200 | ~28 h/cell |
| griewank_10d | 52 → 252 | 200 | **~40 h/cell** |

At 1440 heter_LVGP cells/problem this is **not runnable**. **Measure one real multi-dim heter_LVGP cell
before planning the sweep.** It may force fewer seeds/iterations, a subsampled grid, or a faster fit.

### Probe vehicle: rastrigin_6d — DONE (first real multi-dim problem, both engines)
- [x] Python equations — analytic checks EXACT (f(c(ℓ),ℓ)=b(ℓ) ∀ℓ; lattice identity; σ(c(1),1)=√1.1).
- [x] `problems.m` twin — `verify_problems.py` **ALL MATCH** (max|Δf|=2.8e-14 over 800 random 5-D pts;
      the 4 one-D problems still match through the d-generalized harness).
- [x] Exact ground truth wired via `meta` (`f_star`/`x_star`/`f_star_per_level`) — multi-dim regret is
      analytic, not grid-approximated (uniqueness proved in the source doc).
- [x] **d=5 MATLAB engine smoke: PASS** (heter + standard drivers, 3 iters, valid v2 cells).
      ⚠ Timing: 3 iters = 312 s heter / 158 s standard at n_tr≈33 → **~104 s/iter vs ~9.5 s/iter in 1-D**.
      Even flat extrapolation ⇒ ≥6 h per heter cell; n-growth makes it worse. Full-cell probe running.
- [x] Full 200-iter heter_LVGP cell probe (timestamped): **per-iteration time is FLAT ~41 s over
      n_tr 34→113** — cost is CONSTANT-dominated (6-dim hyperopt + 12k-candidate search), NOT O(n³);
      the 29–40 h/cell projection was wrong. All fits agree: **~2.3–3.6 h per heter cell (≈2.7 h best
      estimate)**. Measured python: separate_gp ≈ 0.3 h/cell @ 200 iters (finds the right basin).

### VERDICT (rastrigin-scale problem, full xlsx budget: 6 acqs × 2 nreps × 30 seeds)
| model | cells | est. core-h |
|---|---|---|
| heter_LVGP | 360 | ~970 |
| standard_LVGP | 180 | ~250 |
| categorical_kernel | 360 | ~350 (TBD, cells running) |
| separate_gp | 360 | ~110 |
⇒ **~1700 core-h ≈ 4 days wall on 18 workers PER PROBLEM**; ×6 problems (10-D ones ~1.5×) ≈
**~4 weeks continuous**. Feasible but long. Rescoping levers (multiplicative): seeds 30→15 (×½),
nrep {3,10}→{10} (×½), heter restricted to noise-aware acqs (heter block ×½), MATLAB num_iter
200→100 (MATLAB blocks ×½, deviates from xlsx). Decision = owner's.

---

## ✅ PHASE 2c — the 6 equations (both engines) — COMPLETE

- [x] All 6 implemented in Python + MATLAB. **`verify_problems.py`: ALL 10 MATCH** (max|Δf| ≤ 9e-13,
      relative ~1e-16; 800 random d-dim points per problem).
- [x] rastrigin_6d (per-level centers) done in 2b. TP-5/6: level value enters as x10; analytic optima
      at x_q=0 wired into meta (exact formula values — the doc's f-min table has small rounding
      artifacts from pre-rounded cosines; we follow the FORMULA).
- [x] ENG-1/2/3: per-variable ranges; ground truth = **exact corner optima** (f monotone over the box;
      256-start Sobol+L-BFGS all converge to the same corner, zero spread) embedded in meta.
      ⚠ Design note: piston/OTL per-level f* gaps (1e-3..7e-3) are BELOW their sigma (up to 0.03/0.1)
      → the intended challenge is LEVEL identification under noise; golinski gaps (~270) are easy.
- [x] DOE caches build for all 5 (X (n, d+1), balanced, in-bounds); d=9 end-to-end python BO smoke
      passes (both models, rahbo, r_at_est finite).
**ALL 10 PROBLEMS LIVE.** Remaining before a full multi-dim sweep: the SCOPE DECISION (Phase 2b verdict
table) — full budget ≈ 4 weeks vs rescoped.

---

## PHASE 3 — analysis gaps (noted, not blocking)
- [ ] n_rep 3 vs 10 comparison (data exists; the "do more replicates help?" question is unasked).
- [ ] Significance testing in the main tables (only `head_to_head` has it).
- [ ] Cost-vs-benefit: heter_LVGP is ~20× slower than separate_gp — is it worth it?
- [ ] `_panel`'s `label=` is commented out → convergence panels render without legends.
