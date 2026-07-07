"""
problem.py -- the heteroscedastic Branin-like mixed-variable test problem.

This is the SAME problem solved by the MATLAB LVGP study (study_v2/study_driver.m):
a 1-D continuous variable x1 in [-5, 10] crossed with a 5-level categorical variable.
The 5 levels map to the Branin second-coordinate VALUES `VAR_FCTR = [15, 2, 8, 0, 10]`,
and each level has its own heteroscedastic-noise multiplier.

Keeping this identical to study_v2 is what makes the per-category-GP study directly
comparable to the LVGP study. The objective / noise / ground-truth / config-metadata
helpers are ported verbatim from study_v2/utils/problem.py; the simulation helpers
(`noisy_eval`, `initial_doe`) are new -- in study_v2 they lived inside study_driver.m.

NOTE on seeds: numpy RNG != MATLAB RNG, so a given seed does NOT reproduce the MATLAB
DOE byte-for-byte. We replicate the *protocol* (2 maximin-LHS x1 locations shared across
all categories, on a 1/6 inset; n_rep noisy replicates per location) and average over 30
seeds -- the comparison is statistically fair, not seed-identical.
"""
import numpy as np

# ---- domain / problem constants (match study_driver.m) ----
LB, UB = -5.0, 10.0
N_LV = 5                                                      # number of categorical levels
VAR_FCTR = np.array([15, 2, 8, 0, 10.])                      # actual Branin x2 value of each level
NOISE_MULS = np.array([1.00, 0.70, 0.90, 0.50, 1.20]) * 10   # per-level noise multiplier
LEVELS = np.arange(1, N_LV + 1)                              # 1-based level indices, as in MATLAB

# initial design-of-experiments protocol (study_driver.m v2)
N_TR_LV = 2                                                  # LHS points per category
EDGE_BUF = 1.0 / 6.0                                         # 1/6 inset each side -> x1 in [-2.5, 7.5]

# Canonical 12-config sweep (acf, param); folder tag matches run_sweep.acf_tag().
CONFIG_ORDER = [
    ("lcb", float("nan")), ("pi", float("nan")), ("ei", float("nan")),
    ("haei", 0.5), ("haei", 1.0), ("haei", 5.0),
    ("anpei", 0.2), ("anpei", 0.5), ("anpei", 0.8),
    ("rahbo", 0.5), ("rahbo", 1.0), ("rahbo", 5.0),
]
_KNOB = {"haei": "g", "rahbo": "a", "anpei": "b"}


# ---- noise-free objective and noise model ----
def f_true(x1, x2):
    """True noise-free objective (Branin-like). x2 is the level VALUE (not index)."""
    x1 = np.asarray(x1, float)
    return ((x2 - 5.1/(4*np.pi**2)*x1**2 + 5/np.pi*x1 - 6)**2
            + 10*(1 - 1/(8*np.pi))*np.cos(x1) + 10)


def f_true_level(x1, level):
    """True noise-free objective at continuous x1 and 1-based categorical `level`."""
    return f_true(x1, VAR_FCTR[level - 1])


def sigma(x1, level):
    """Heteroscedastic noise std at continuous x1 and 1-based categorical level.
    base_sigma(x1) = 0.135*exp((0.15*x1)^2); multiplied by the per-level factor."""
    x1 = np.asarray(x1, float)
    return 0.135*np.exp((0.15*x1)**2) * NOISE_MULS[level - 1]


# ---- stochastic simulation (the BO oracle) ----
def noisy_eval(x1, level, n_rep, rng):
    """Evaluate the noisy objective `n_rep` times at (x1, level).
    Returns the length-n_rep replicate vector y = f_true + N(0, sigma^2)."""
    f = f_true_level(x1, level)
    s = sigma(x1, level)
    return f + rng.standard_normal(n_rep) * s


def _maximin_lhs_1d(rng, n, n_iter=8000):
    """maximin Latin-hypercube sample of `n` points in [0,1] (one per stratum), emulating
    MATLAB lhsdesign's default 'maximin' criterion (maximize the min pairwise distance over
    many seeded LHS draws). For n=2 this pins the points to the unit-interval edges -> the
    v2 design's tight, seed-consistent bands. n_iter=8000 reproduces study_v2's DOE on BOTH
    edges (x1 ~ [-2.45, 7.46] every seed). Returns the sorted points."""
    edges = np.linspace(0.0, 1.0, n + 1)
    lo_e, hi_e = edges[:-1], edges[1:]
    # vectorized: draw all n_iter candidate LHS designs at once (one point per stratum),
    # then keep the design with the largest minimum pairwise gap. ~1000x faster than a
    # Python loop over n_iter (numpy overhead on 2-element arrays dominates the loop).
    cand = rng.uniform(lo_e, hi_e, size=(n_iter, n))         # (n_iter, n)
    cand.sort(axis=1)
    gaps = np.diff(cand, axis=1).min(axis=1) if n > 1 else np.ones(n_iter)
    return cand[int(gaps.argmax())]                          # design maximizing the min gap


def initial_doe(n_rep, seed=None, rng=None):
    """Build the shared-LHS replicated initial design, mirroring study_driver.m.

    2 maximin-LHS x1 locations on the 1/6 inset [-2.5, 7.5] are generated ONCE and
    SHARED across all 5 categories; every (location, level) pair is evaluated with
    `n_rep` noisy replicates. Returns a dict of numpy arrays:
        X_sample   (n_tr, 2)  columns [x1, level_idx(1-based)]
        Y_sample   (n_tr,)    replicate mean at each location
        Var_sample (n_tr,)    replicate (sample) variance at each location
        Y_rep      (n_tr, n_rep) raw replicates
    with n_tr = N_LV * N_TR_LV = 10.

    Pass an existing `rng` (np.random.Generator) so the BO loop can continue the SAME
    noise stream afterwards (one rng(seed) controls DOE + all in-loop noise, as in MATLAB).
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    lo = LB + EDGE_BUF * (UB - LB)                            # -2.5  (center of lower third)
    hi = UB - EDGE_BUF * (UB - LB)                            #  7.5  (center of upper third)
    # maximin LHS for the 2 shared x1 locations, matching MATLAB lhsdesign's default
    # 'maximin' criterion. (scipy qmc's random-cd minimizes DISCREPANCY, not min-distance,
    # so it does NOT pin to the edges -> a dispersed, seed-varying DOE unlike study_v2.)
    # For 2 points maximin pins them to the inset edges ~[-2.5, 7.5] with tight,
    # seed-consistent bands, reproducing study_v2's DOE (x1 ~ [-2.45, 7.47] every seed).
    A = _maximin_lhs_1d(rng, N_TR_LV, n_iter=8000)
    lhs_shared = A * (hi - lo) + lo                          # same x1 values for every level

    n_tr = N_LV * N_TR_LV
    X_sample = np.zeros((n_tr, 2))
    Y_sample = np.zeros(n_tr)
    Var_sample = np.zeros(n_tr)
    Y_rep = np.zeros((n_tr, n_rep))
    row = 0
    for i in range(1, N_LV + 1):                             # 1-based level, as in MATLAB
        for j in range(N_TR_LV):
            y_rep = noisy_eval(lhs_shared[j], i, n_rep, rng)
            X_sample[row] = [lhs_shared[j], i]
            Y_sample[row] = y_rep.mean()
            Var_sample[row] = y_rep.var(ddof=1)              # unbiased, matches MATLAB var(...,0)
            Y_rep[row] = y_rep
            row += 1
    return dict(X_sample=X_sample, Y_sample=Y_sample,
                Var_sample=Var_sample, Y_rep=Y_rep, lhs_shared=lhs_shared)


# ---- ground-truth summaries (ported from study_v2) ----
def ground_truth_min():
    """True noise-free global minimum over x1 in [LB, UB] and all levels."""
    x1 = np.linspace(LB, UB, 4000)
    return float(min(f_true(x1, v).min() for v in VAR_FCTR))


def true_opt_location():
    """(level, x1) of the true global optimum."""
    x1 = np.linspace(LB, UB, 4000)
    best = (np.inf, None, None)
    for lv, v in enumerate(VAR_FCTR, 1):
        fv = f_true(x1, v)
        if fv.min() < best[0]:
            best = (fv.min(), lv, x1[fv.argmin()])
    return best[1], best[2]                                  # level, x1


def true_min_per_category():
    """True noise-free minimum of f within each category level."""
    x1 = np.linspace(LB, UB, 4000)
    return np.array([f_true(x1, v).min() for v in VAR_FCTR])


# ---- config labels / tags / canonical keys (ported from study_v2) ----
def label(acf, param):
    if acf == "haei":  return f"HAEI(γ={param:g})"
    if acf == "rahbo": return f"RAHBO(α={param:g})"
    if acf == "anpei": return f"ANPEI(β={param:g})"
    return acf.upper()


def acf_tag(acf, param):
    """Folder tag for an acquisition config (matches run_sweep.acf_tag())."""
    if param != param:                                       # NaN -> no knob (ei/lcb/pi)
        return acf
    return f"{acf}_{_KNOB.get(acf, 'p')}{param:g}"


def canon_cfg(acf, param):
    """Canonical config key: (acf, 'na' if NaN else rounded param)."""
    return (acf, "na" if param != param else round(float(param), 6))


# ---- journal-level styling (ported from study_v2): color+marker=family, linestyle=param ----
FAMILY_COLOR = {
    "lcb": "#0072B2", "pi": "#E69F00", "ei": "#009E73",
    "haei": "#D55E00", "anpei": "#CC79A7", "rahbo": "#56B4E9",
}
FAMILY_MARKER = {
    "lcb": "v", "pi": "s", "ei": "o", "haei": "^", "anpei": "D", "rahbo": "P",
}
_LINESTYLES = ["-", "--", ":", "-."]
_FAMILY_PARAMS = {}
for _a, _p in CONFIG_ORDER:
    if _p == _p:
        _FAMILY_PARAMS.setdefault(_a, set()).add(round(float(_p), 6))
_FAMILY_PARAMS = {a: sorted(v) for a, v in _FAMILY_PARAMS.items()}


def style_for(acf, param):
    """(color, marker, linestyle) for a config: color+marker=family, linestyle=param rank."""
    color = FAMILY_COLOR.get(acf, "#444444")
    marker = FAMILY_MARKER.get(acf, "o")
    if param != param:
        ls = "-"
    else:
        params = _FAMILY_PARAMS.get(acf, [])
        p = round(float(param), 6)
        rank = params.index(p) if p in params else 0
        ls = _LINESTYLES[rank % len(_LINESTYLES)]
    return {"color": color, "marker": marker, "linestyle": ls}


if __name__ == "__main__":
    lv, x = true_opt_location()
    print(f"true global min = {ground_truth_min():.6g} at level {lv} (x2={VAR_FCTR[lv-1]:g}), x1={x:.4f}")
    print("true min per category:", np.round(true_min_per_category(), 4))
    doe = initial_doe(n_rep=10, seed=1)
    print("DOE shared x1 locations:", np.round(doe["lhs_shared"], 4))
    print("DOE X_sample[:,1] (levels):", doe["X_sample"][:, 1].astype(int))
