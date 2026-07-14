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

---

## PHASE 2c — the 6 equations (both engines)

- [ ] Python `utils/problems.py` + MATLAB `matlab/problems.m` (⚠ `reshape(level_lookup, size(x1))` —
      MATLAB `v(lv)` keeps v's row orientation and silently broadcasts to a matrix).
- [ ] `verify_problems.py` — Python == MATLAB to ~1e-15 for all 10 (as done for the 1-D four).
- [ ] ⚠ **rastrigin_6d** is special: per-level **centers** c(ℓ) shift the optimum *location*, not just noise.
- [ ] Engineering problems (golinski / piston / otl_circuit) have **per-variable ranges** — already in
      `doe.PROBLEM_GRID`.

---

## PHASE 3 — analysis gaps (noted, not blocking)
- [ ] n_rep 3 vs 10 comparison (data exists; the "do more replicates help?" question is unasked).
- [ ] Significance testing in the main tables (only `head_to_head` has it).
- [ ] Cost-vs-benefit: heter_LVGP is ~20× slower than separate_gp — is it worth it?
- [ ] `_panel`'s `label=` is commented out → convergence panels render without legends.
