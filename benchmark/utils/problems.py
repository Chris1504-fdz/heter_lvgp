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

from . import doe as _doe                                # SLHD / space-filling DOE generators


@dataclass
class ProblemSpec:
    """One mixed-variable test problem: d CONTINUOUS dims + 1 categorical.
      f     : f(X, level_1based) -> noise-free objective;  X has shape (n, d), returns (n,)
      sigma : sigma(X, level_1based) -> heteroscedastic noise std, same shapes
      bounds: [(lb, ub)] per continuous dim.  For a 1-D problem you may pass scalar lb/ub instead.
      n_levels : number of categorical levels (1-based 1..n_levels)
      n_init : TOTAL initial design points (all levels);  num_iter : BO iterations
    n_init/num_iter default to this problem's row of doe.PROBLEM_GRID (= resources/init_doe_iter.xlsx).

    `lb`/`ub` remain as scalars for the 1-D problems (and are derived from bounds[0] otherwise), so
    every existing d=1 call site keeps working unchanged."""
    name: str
    f: Callable
    sigma: Callable
    lb: float = None                                 # 1-D convenience (or derived from bounds[0])
    ub: float = None
    n_levels: int = 0
    n_init: int = 0                                  # 0 -> take from PROBLEM_GRID
    num_iter: int = 0                                # 0 -> take from PROBLEM_GRID
    bounds: object = None                            # [(lb,ub)] per dim; None -> [(lb, ub)] (d=1)
    edge_buf: float = 1.0 / 6.0
    meta: dict = field(default_factory=dict)         # optional: cat_values, noise_muls, notes

    def __post_init__(self):
        if self.bounds is None:
            if self.lb is None or self.ub is None:
                raise ValueError(f"{self.name}: give bounds=[(lb,ub),...] or scalar lb/ub")
            self.bounds = [(float(self.lb), float(self.ub))]
        self.bounds = np.atleast_2d(np.asarray(self.bounds, float))     # (d, 2)
        if self.lb is None:                          # keep the 1-D convenience fields consistent
            self.lb, self.ub = float(self.bounds[0, 0]), float(self.bounds[0, 1])
        g = _doe.PROBLEM_GRID.get(self.name)
        if not self.n_init:
            self.n_init = g.n_init if g else 2 * self.n_levels
        if not self.num_iter:
            self.num_iter = g.num_iter if g else 50

    @property
    def d(self):
        """Number of CONTINUOUS dimensions."""
        return int(self.bounds.shape[0])

    @property
    def n_tr_lv(self):
        """Design points per level (floor). `n_init % n_levels` levels get one extra -- see doe.make_doe."""
        return self.n_init // self.n_levels

    def _farg(self, X):
        """Argument actually passed to f/sigma.
        d==1: `np.asarray(X)` UNCHANGED -- byte-for-byte the original 1-D behaviour (0-d for a scalar,
              (n,) for an array), so the existing f/sigma stay untouched and bit-identical.
        d>1 : the full (n, d) array; a single 1-D point is promoted to (1, d)."""
        X = np.asarray(X, float)
        if self.d == 1:
            return X                                   # exactly what the old f_true_level passed
        return X.reshape(1, self.d) if X.ndim == 1 else X.reshape(-1, self.d)

    # convenience
    def f_true_level(self, X, level):
        return np.asarray(self.f(self._farg(X), int(level)), float)

    def sigma_level(self, X, level):
        return np.asarray(self.sigma(self._farg(X), int(level)), float)

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
#  TP-2  Six-Hump Camel-Back (redesigned) -- 1 continuous + 4 levels
# ======================================================================================
_CAMEL_VALS = [0.2, 0.4, 0.7, 1.0]
_CAMEL_MULS = [2.0, 3.5, 1.5, 5.0]
def _camel_f(x1, level):
    x1 = np.asarray(x1, float); x2 = _CAMEL_VALS[level - 1]
    return (4 - 2.1 * x1 ** 2 + x1 ** 4 / 3) * x1 ** 2 + x1 * x2 + (-4 + 4 * x2 ** 2) * x2 ** 2
def _camel_sigma(x1, level):
    x1 = np.asarray(x1, float)
    return 0.05 * np.exp((0.4 * x1) ** 2) * _CAMEL_MULS[level - 1]
CAMEL = ProblemSpec("sixhump_camel", _camel_f, _camel_sigma, -2.0, 2.0, 4,
                    meta=dict(cat_values=_CAMEL_VALS, noise_muls=_CAMEL_MULS))

# ======================================================================================
#  TP-3  Griewank 2-D (redesigned) -- 1 continuous + 4 levels
# ======================================================================================
_GRIE_VALS = [0.0, 0.5, 1.0, 1.5]
_GRIE_MULS = [2.0, 1.0, 3.5, 1.5]
def _griewank2d_f(x1, level):
    x1 = np.asarray(x1, float); x2 = _GRIE_VALS[level - 1]
    return (x1 ** 2 + x2 ** 2) / 4000 - np.cos(x1) * np.cos(x2 / np.sqrt(2)) + 1
def _griewank2d_sigma(x1, level):
    x1 = np.asarray(x1, float)
    return 0.04 * (1 + 0.08 * x1 ** 2) * _GRIE_MULS[level - 1]
GRIEWANK2D = ProblemSpec("griewank_2d", _griewank2d_f, _griewank2d_sigma, -5.0, 5.0, 4,
                         meta=dict(cat_values=_GRIE_VALS, noise_muls=_GRIE_MULS))

# ======================================================================================
#  TP-4  Ackley 2-D (replaces Goldstein-Price) -- 1 continuous + 4 levels
# ======================================================================================
_ACK_VALS = [0.0, 0.5, 1.5, 2.5]
_ACK_MULS = [1.0, 2.0, 1.5, 3.0]
def _ackley2d_f(x1, level):
    x1 = np.asarray(x1, float); x2 = _ACK_VALS[level - 1]; a, b, c = 20.0, 0.2, 2 * np.pi
    return (-a * np.exp(-b * np.sqrt((x1 ** 2 + x2 ** 2) / 2))
            - np.exp((np.cos(c * x1) + np.cos(c * x2)) / 2) + a + np.e)
def _ackley2d_sigma(x1, level):
    x1 = np.asarray(x1, float)
    return 0.10 * (1 + 0.15 * x1 ** 2) * _ACK_MULS[level - 1]
ACKLEY2D = ProblemSpec("ackley_2d", _ackley2d_f, _ackley2d_sigma, -3.0, 3.0, 4,
                       meta=dict(cat_values=_ACK_VALS, noise_muls=_ACK_MULS))

# ======================================================================================
#  Phase-2 multi-dim problems (TP-5/6/7, ENG-1/2/3) -- stubs until the framework is
#  generalized to d continuous dimensions.
# ======================================================================================
def _todo(name):
    def _f(x1, level):
        raise NotImplementedError(f"problem '{name}' not defined yet -- fill in problems.py")
    return _f


def _stub(name, n_levels=5, lb=-5.0, ub=10.0):
    return ProblemSpec(name=name, f=_todo(name), sigma=_todo(name), lb=lb, ub=ub, n_levels=n_levels)


# Registry. 1-D problems are live; the 6 multi-dim ones are stubs until the d-dim generalization.
PROBLEMS = {
    "branin_hetero": BRANIN,        # TP-1  (1-D, 5 levels)
    "sixhump_camel": CAMEL,         # TP-2  (1-D, 4 levels)
    "griewank_2d":   GRIEWANK2D,    # TP-3  (1-D, 4 levels)
    "ackley_2d":     ACKLEY2D,      # TP-4  (1-D, 4 levels)
    "griewank_10d":  _stub("griewank_10d"),   # TP-5  (9-D)  Phase 2
    "ackley_10d":    _stub("ackley_10d"),      # TP-6  (9-D)  Phase 2
    "rastrigin_6d":  _stub("rastrigin_6d"),    # TP-7  (5-D)  Phase 2
    "golinski":      _stub("golinski"),        # ENG-1 (6-D)  Phase 2
    "piston":        _stub("piston"),          # ENG-2 (6-D)  Phase 2
    "otl_circuit":   _stub("otl_circuit"),     # ENG-3 (5-D)  Phase 2
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
def noisy_eval(spec, x, level, n_rep, rng):
    """n_rep noisy replicates of the objective at (x, level): f + N(0, sigma^2).
    `x` is a scalar (d=1) or a length-d vector -- one design point."""
    f = float(np.ravel(spec.f_true_level(x, level))[0])
    s = float(np.ravel(spec.sigma_level(x, level))[0])
    return f + rng.standard_normal(n_rep) * s


def _maximin_lhs_1d(rng, n, n_iter=8000):
    """maximin LHS of n points in [0,1] (emulates MATLAB lhsdesign default), vectorized."""
    edges = np.linspace(0.0, 1.0, n + 1)
    lo_e, hi_e = edges[:-1], edges[1:]
    cand = rng.uniform(lo_e, hi_e, size=(n_iter, n))
    cand.sort(axis=1)
    gaps = np.diff(cand, axis=1).min(axis=1) if n > 1 else np.ones(n_iter)
    return cand[int(gaps.argmax())]


DOE_MODE = "slhd"                                       # benchmark default (see notebooks/doe.ipynb)


def initial_doe(spec, n_rep, seed=None, rng=None, mode=DOE_MODE, n_init=None):
    """Replicated initial design of spec.n_init TOTAL points. Default = SLHD over the FULL domain
    [lb,ub]: each level is an LHS AND the union is a full LHS (see notebooks/doe.ipynb). If n_init is
    not a multiple of n_levels, `n_init % n_levels` levels get one extra point.
    Each design point is evaluated n_rep times (heteroscedastic replicates). Both engines consume this
    via utils/doe_cache.py, so the MATLAB drivers cannot drift from it.
    Returns: X_sample (n_tr, d+1) = [x_1..x_d, level], Y_sample (replicate mean), Var_sample
    (replicate var), Y_rep (n_tr,n_rep), doe {level: (m, d) array}.
    d == 1 is BIT-EXACT with the pre-d-dim code: make_doe_nd consumes identical rng there (the dim
    shuffle is skipped), so every design location and noise draw is unchanged."""
    if rng is None:
        rng = np.random.default_rng(seed)
    n_tr = int(spec.n_init if n_init is None else n_init)
    d = spec.d
    doe = _doe.make_doe_nd(mode, rng, spec.bounds, spec.n_levels, n_init=n_tr)
    X_sample = np.zeros((n_tr, d + 1)); Y_sample = np.zeros(n_tr)
    Var_sample = np.zeros(n_tr); Y_rep = np.zeros((n_tr, n_rep))
    row = 0
    for i in range(1, spec.n_levels + 1):
        for xrow in doe[i]:                             # this level's design points, each (d,)
            y_rep = noisy_eval(spec, xrow, i, n_rep, rng)
            X_sample[row, :d] = xrow; X_sample[row, d] = i
            Y_sample[row] = y_rep.mean()
            Var_sample[row] = y_rep.var(ddof=1)
            Y_rep[row] = y_rep
            row += 1
    return dict(X_sample=X_sample, Y_sample=Y_sample, Var_sample=Var_sample,
                Y_rep=Y_rep, doe=doe)


def _grid_1d_only(spec, fn_name):
    if spec.d > 1:
        raise NotImplementedError(
            f"{fn_name}: dense-grid ground truth is 1-D only; {spec.name} has d={spec.d}. "
            "Phase 2c will add a Sobol+polish version for the multi-dim problems.")


def ground_truth_min(spec, n=4000):
    _grid_1d_only(spec, "ground_truth_min")
    x1 = np.linspace(spec.lb, spec.ub, n)
    return float(min(spec.f_true_level(x1, lv).min() for lv in spec.levels))


def true_opt_location(spec, n=4000):
    _grid_1d_only(spec, "true_opt_location")
    x1 = np.linspace(spec.lb, spec.ub, n)
    best = (np.inf, None, None)
    for lv in spec.levels:
        fv = spec.f_true_level(x1, lv)
        if fv.min() < best[0]:
            best = (fv.min(), lv, x1[fv.argmin()])
    return best[1], best[2]


def true_min_per_category(spec, n=4000):
    _grid_1d_only(spec, "true_min_per_category")
    x1 = np.linspace(spec.lb, spec.ub, n)
    return np.array([spec.f_true_level(x1, lv).min() for lv in spec.levels])
