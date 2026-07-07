"""
problems.py -- registry of the mixed-variable (1-D continuous x1 + K-level categorical) test
problems. Each problem is a ProblemSpec giving the noise-free objective f(x1, level) and the
heteroscedastic noise std sigma(x1, level); everything else (DOE, ground truth, regret) is
computed generically from the spec, so ADDING A FUNCTION = one ProblemSpec entry.

`branin_hetero` is the worked reference (identical to study_v2/study_driver.m and
study_v2_gp/utils/problem.py). The other 8 are stubs -- drop the ground-truth equations in.
IMPORTANT: each function must also be mirrored in matlab/problems.m (verify_problems.py checks
they agree), because the two LVGP models run in MATLAB.
"""
from dataclasses import dataclass, field
from typing import Callable, List
import numpy as np


@dataclass
class ProblemSpec:
    """One mixed-variable test problem.
      f     : f(x1_array, level_1based) -> noise-free objective (numpy in/out)
      sigma : sigma(x1_array, level_1based) -> heteroscedastic noise std
      lb,ub : continuous domain of x1
      n_levels : number of categorical levels (1-based 1..n_levels)
    DOE knobs match study_driver.m (2 maximin-LHS x1 locations on a 1/6 inset, shared across levels)."""
    name: str
    f: Callable
    sigma: Callable
    lb: float
    ub: float
    n_levels: int
    n_tr_lv: int = 2
    edge_buf: float = 1.0 / 6.0
    meta: dict = field(default_factory=dict)         # optional: cat_values, noise_muls, notes

    # convenience
    def f_true_level(self, x1, level):
        return np.asarray(self.f(np.asarray(x1, float), int(level)), float)

    def sigma_level(self, x1, level):
        return np.asarray(self.sigma(np.asarray(x1, float), int(level)), float)

    @property
    def levels(self):
        return list(range(1, self.n_levels + 1))


# ======================================================================================
#  WORKED REFERENCE: heteroscedastic Branin (== study_v2 / study_v2_gp)
# ======================================================================================
_BRANIN_VAR_FCTR = np.array([15, 2, 8, 0, 10.0])                 # Branin x2 value per level
_BRANIN_NOISE_MULS = np.array([1.00, 0.70, 0.90, 0.50, 1.20]) * 10


def _branin_raw(x1, x2):
    x1 = np.asarray(x1, float)
    return ((x2 - 5.1 / (4 * np.pi ** 2) * x1 ** 2 + 5 / np.pi * x1 - 6) ** 2
            + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1) + 10)


def _branin_f(x1, level):
    return _branin_raw(x1, _BRANIN_VAR_FCTR[level - 1])


def _branin_sigma(x1, level):
    x1 = np.asarray(x1, float)
    return 0.135 * np.exp((0.15 * x1) ** 2) * _BRANIN_NOISE_MULS[level - 1]


BRANIN = ProblemSpec(
    name="branin_hetero", f=_branin_f, sigma=_branin_sigma, lb=-5.0, ub=10.0, n_levels=5,
    meta=dict(cat_values=_BRANIN_VAR_FCTR.tolist(), noise_muls=_BRANIN_NOISE_MULS.tolist()),
)


# ======================================================================================
#  8 stubs -- replace `f`/`sigma`/`lb`/`ub`/`n_levels` with the user's ground-truth equations,
#  then mirror each in matlab/problems.m.  Registered below only when filled in.
# ======================================================================================
def _todo(name):
    def _f(x1, level):
        raise NotImplementedError(f"problem '{name}' not defined yet -- fill in problems.py")
    return _f


def _stub(name, n_levels=5, lb=-5.0, ub=10.0):
    return ProblemSpec(name=name, f=_todo(name), sigma=_todo(name), lb=lb, ub=ub, n_levels=n_levels)


# Registry. Add the 8 as you supply equations (replace _stub(...) with a real ProblemSpec).
PROBLEMS = {
    "branin_hetero": BRANIN,
    "fn2": _stub("fn2"), "fn3": _stub("fn3"), "fn4": _stub("fn4"), "fn5": _stub("fn5"),
    "fn6": _stub("fn6"), "fn7": _stub("fn7"), "fn8": _stub("fn8"), "fn9": _stub("fn9"),
}


def get(name) -> ProblemSpec:
    if name not in PROBLEMS:
        raise KeyError(f"unknown problem '{name}'. available: {list(PROBLEMS)}")
    return PROBLEMS[name]


def defined_problems():
    """Names whose equations are actually implemented (stubs excluded)."""
    out = []
    for nm, sp in PROBLEMS.items():
        try:
            sp.f_true_level(np.array([0.0]), 1)
            out.append(nm)
        except NotImplementedError:
            pass
    return out


# ======================================================================================
#  spec-driven simulation / DOE / ground truth (generic over any ProblemSpec)
# ======================================================================================
def noisy_eval(spec, x1, level, n_rep, rng):
    """n_rep noisy replicates of the objective at (x1, level): f + N(0, sigma^2)."""
    f = float(spec.f_true_level(x1, level))
    s = float(spec.sigma_level(x1, level))
    return f + rng.standard_normal(n_rep) * s


def _maximin_lhs_1d(rng, n, n_iter=8000):
    """maximin LHS of n points in [0,1] (emulates MATLAB lhsdesign default), vectorized."""
    edges = np.linspace(0.0, 1.0, n + 1)
    lo_e, hi_e = edges[:-1], edges[1:]
    cand = rng.uniform(lo_e, hi_e, size=(n_iter, n))
    cand.sort(axis=1)
    gaps = np.diff(cand, axis=1).min(axis=1) if n > 1 else np.ones(n_iter)
    return cand[int(gaps.argmax())]


def initial_doe(spec, n_rep, seed=None, rng=None):
    """Shared-LHS replicated initial design (mirrors study_driver.m). Returns dict of arrays:
    X_sample (n_tr,2)[x1,level], Y_sample (mean), Var_sample (var), Y_rep (n_tr,n_rep), lhs_shared."""
    if rng is None:
        rng = np.random.default_rng(seed)
    lo = spec.lb + spec.edge_buf * (spec.ub - spec.lb)
    hi = spec.ub - spec.edge_buf * (spec.ub - spec.lb)
    A = _maximin_lhs_1d(rng, spec.n_tr_lv, n_iter=8000)
    lhs_shared = A * (hi - lo) + lo
    n_tr = spec.n_levels * spec.n_tr_lv
    X_sample = np.zeros((n_tr, 2)); Y_sample = np.zeros(n_tr)
    Var_sample = np.zeros(n_tr); Y_rep = np.zeros((n_tr, n_rep))
    row = 0
    for i in range(1, spec.n_levels + 1):
        for j in range(spec.n_tr_lv):
            y_rep = noisy_eval(spec, lhs_shared[j], i, n_rep, rng)
            X_sample[row] = [lhs_shared[j], i]
            Y_sample[row] = y_rep.mean()
            Var_sample[row] = y_rep.var(ddof=1)
            Y_rep[row] = y_rep
            row += 1
    return dict(X_sample=X_sample, Y_sample=Y_sample, Var_sample=Var_sample,
                Y_rep=Y_rep, lhs_shared=lhs_shared)


def ground_truth_min(spec, n=4000):
    x1 = np.linspace(spec.lb, spec.ub, n)
    return float(min(spec.f_true_level(x1, lv).min() for lv in spec.levels))


def true_opt_location(spec, n=4000):
    x1 = np.linspace(spec.lb, spec.ub, n)
    best = (np.inf, None, None)
    for lv in spec.levels:
        fv = spec.f_true_level(x1, lv)
        if fv.min() < best[0]:
            best = (fv.min(), lv, x1[fv.argmin()])
    return best[1], best[2]


def true_min_per_category(spec, n=4000):
    x1 = np.linspace(spec.lb, spec.ub, n)
    return np.array([spec.f_true_level(x1, lv).min() for lv in spec.levels])
