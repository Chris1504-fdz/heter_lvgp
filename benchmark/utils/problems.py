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
#  TP-7  Level-Shifted Rastrigin 6-D -- 5 continuous + 4 levels. Each level's global minimum
#  is at a DISTINCT location c(l) with a DISTINCT value b(l) (proved unique in the doc):
#     f(xq, l) = sum_i [(x_i - c_i(l))^2 - 10 cos(2 pi (x_i - c_i(l))) + 10] + b(l)
#     sigma(xq, l) = sqrt(1 + 0.05 mean(xq^2)) * mult(l)
# ======================================================================================
_RAST_C = np.array([[-2.,  1., -1.,  2.,  0.],      # level 1 (a)
                    [ 1., -2.,  2., -1., -2.],      # level 2 (b)
                    [ 2.,  2., -2.,  0.,  1.],      # level 3 (c)
                    [-1., -1.,  0., -2.,  2.]])     # level 4 (d)
_RAST_B = np.array([5., 20., 40., 65.])
_RAST_MULS = np.array([1.0, 2.0, 1.5, 3.0])


def _rastrigin6d_f(X, level):
    D = X - _RAST_C[level - 1]
    return np.sum(D ** 2 - 10.0 * np.cos(2.0 * np.pi * D) + 10.0, axis=1) + _RAST_B[level - 1]


def _rastrigin6d_sigma(X, level):
    return np.sqrt(1.0 + 0.05 * np.mean(X ** 2, axis=1)) * _RAST_MULS[level - 1]


RASTRIGIN6D = ProblemSpec(
    "rastrigin_6d", _rastrigin6d_f, _rastrigin6d_sigma, bounds=[(-5.0, 5.0)] * 5, n_levels=4,
    meta=dict(centers=_RAST_C.tolist(), b_shift=_RAST_B.tolist(), noise_muls=_RAST_MULS.tolist(),
              # analytic ground truth (unique; see the doc's uniqueness proof)
              f_star=5.0, opt_level=1, x_star=_RAST_C[0].tolist(),
              f_star_per_level=_RAST_B.tolist()),
)


# ======================================================================================
#  TP-5  Griewank 10-D -- 9 continuous + 4 levels (level value v enters as x10):
#     f(xq, l) = sum_{i=1..10} xs_i^2/4000 - prod_{i=1..10} cos(xs_i/sqrt(i)) + 1,  xs=[xq, v(l)]
#     sigma    = 0.05*sqrt(1 + 0.1*||xq||^2/9) * mult(l)
#  Unique global min at xq = 0 for every level (all cos(v/sqrt(10)) > 0; doc's level redesign).
# ======================================================================================
_G10_VALS = np.array([1.0, 2.0, 3.0, 4.0])
_G10_MULS = np.array([1.5, 1.0, 3.0, 2.0])
_G10_SQRT_I = np.sqrt(np.arange(1, 11, dtype=float))


def _griewank10d_f(X, level):
    v = _G10_VALS[level - 1]
    XS = np.hstack([X, np.full((X.shape[0], 1), v)])
    return np.sum(XS ** 2, axis=1) / 4000.0 - np.prod(np.cos(XS / _G10_SQRT_I), axis=1) + 1.0


def _griewank10d_sigma(X, level):
    return 0.05 * np.sqrt(1.0 + 0.1 * np.sum(X ** 2, axis=1) / 9.0) * _G10_MULS[level - 1]


_G10_FSTAR = [float(v * v / 4000.0 + 1.0 - np.cos(v / np.sqrt(10.0))) for v in _G10_VALS]
GRIEWANK10D = ProblemSpec(
    "griewank_10d", _griewank10d_f, _griewank10d_sigma, bounds=[(-5.0, 5.0)] * 9, n_levels=4,
    meta=dict(cat_values=_G10_VALS.tolist(), noise_muls=_G10_MULS.tolist(),
              f_star=_G10_FSTAR[0], opt_level=1, x_star=[0.0] * 9, f_star_per_level=_G10_FSTAR),
)


# ======================================================================================
#  TP-6  Ackley 10-D -- 9 continuous + 4 levels (level value v enters as x10), a=20 b=0.2 c=2pi:
#     f(xq, l) = -20 exp(-0.2 sqrt(sum(xs^2)/10)) - exp(sum(cos(2 pi xs))/10) + 20 + e
#     sigma    = 0.10*exp(0.20*sqrt(mean(xq^2))) * mult(l)
#  Unique global min at xq = 0 (integer v => cos(2 pi v)=1): f* = 20(1 - exp(-0.2 v/sqrt(10))).
# ======================================================================================
_A10_VALS = np.array([1.0, 2.0, 3.0, 4.0])
_A10_MULS = np.array([2.0, 1.0, 3.5, 1.5])


def _ackley10d_f(X, level):
    v = _A10_VALS[level - 1]
    XS = np.hstack([X, np.full((X.shape[0], 1), v)])
    return (-20.0 * np.exp(-0.2 * np.sqrt(np.sum(XS ** 2, axis=1) / 10.0))
            - np.exp(np.sum(np.cos(2.0 * np.pi * XS), axis=1) / 10.0) + 20.0 + np.e)


def _ackley10d_sigma(X, level):
    return 0.10 * np.exp(0.20 * np.sqrt(np.mean(X ** 2, axis=1))) * _A10_MULS[level - 1]


_A10_FSTAR = [float(20.0 * (1.0 - np.exp(-0.2 * v / np.sqrt(10.0)))) for v in _A10_VALS]
ACKLEY10D = ProblemSpec(
    "ackley_10d", _ackley10d_f, _ackley10d_sigma, bounds=[(-5.0, 5.0)] * 9, n_levels=4,
    meta=dict(cat_values=_A10_VALS.tolist(), noise_muls=_A10_MULS.tolist(),
              f_star=_A10_FSTAR[0], opt_level=1, x_star=[0.0] * 9, f_star_per_level=_A10_FSTAR),
)


# ======================================================================================
#  ENG-1  Golinski speed reducer -- 6 continuous [x1 face-width, x2 module, x4, x5 shaft lengths,
#  x6, x7 diameters] + x3 = teeth count as the 5-level categorical {17,19,21,25,28}:
#     f = 0.7854 x1 x2^2 (3.3333 x3^2 + 14.9334 x3 - 43.0934) - 1.508 x1 (x6^2 + x7^2)
#         + 7.4777 (x6^3 + x7^3) + 0.7854 (x4 x6^2 + x5 x7^2)
#     sigma = 20 exp(0.40(x1-2.6)) (1 + 0.20(x6-2.9)) * mult(l)
#  (X columns: x1, x2, x4, x5, x6, x7 -- same order as PROBLEM_GRID bounds.)
# ======================================================================================
_GOL_VALS = np.array([17.0, 19.0, 21.0, 25.0, 28.0])
_GOL_MULS = np.array([1.2, 1.0, 1.5, 1.8, 1.3])


def _golinski_f(X, level):
    x1, x2, x4, x5, x6, x7 = (X[:, j] for j in range(6))
    x3 = _GOL_VALS[level - 1]
    return (0.7854 * x1 * x2 ** 2 * (3.3333 * x3 ** 2 + 14.9334 * x3 - 43.0934)
            - 1.508 * x1 * (x6 ** 2 + x7 ** 2) + 7.4777 * (x6 ** 3 + x7 ** 3)
            + 0.7854 * (x4 * x6 ** 2 + x5 * x7 ** 2))


def _golinski_sigma(X, level):
    return 20.0 * np.exp(0.40 * (X[:, 0] - 2.6)) * (1.0 + 0.20 * (X[:, 4] - 2.9)) * _GOL_MULS[level - 1]


# Ground truth: f is monotone over the box -> unique BOUNDARY optimum, identical corner for every
# level (verified: 256-start Sobol + L-BFGS all converge there, zero spread). Values = exact corner eval.
_GOL_XSTAR = [2.6, 0.7, 7.3, 7.3, 2.9, 5.0]
_GOL_FSTAR = [2352.44784872076, 2622.474059415, 2919.18265928268, 3592.6470265383596,
              4167.786573560399]
GOLINSKI = ProblemSpec(
    "golinski", _golinski_f, _golinski_sigma, n_levels=5,
    bounds=[(2.6, 3.6), (0.7, 0.8), (7.3, 8.3), (7.3, 8.3), (2.9, 3.9), (5.0, 5.5)],
    meta=dict(cat_values=_GOL_VALS.tolist(), noise_muls=_GOL_MULS.tolist(),
              f_star=_GOL_FSTAR[0], opt_level=1, x_star=_GOL_XSTAR, f_star_per_level=_GOL_FSTAR),
)


# ======================================================================================
#  ENG-2  Piston simulation -- 6 continuous [M, S, V0, k, P0, Ta] + T0 as the 4-level categorical
#  {340,346,352,358} K. Cycle time:
#     A = P0 S + 19.62 M - k V0 / S
#     V = (S / 2k) (sqrt(A^2 + 4 k P0 V0 Ta / T0) - A)
#     f = C = 2 pi sqrt(M / (k + S^2 P0 V0 Ta / (T0 V^2)))
#     sigma = 0.002 (1 + 3 (P0 - 90000)/20000) (1 + 0.5 (M - 30)/30) * mult(l)
# ======================================================================================
_PIS_VALS = np.array([340.0, 346.0, 352.0, 358.0])
_PIS_MULS = np.array([1.0, 1.3, 1.6, 2.0])


def _piston_f(X, level):
    M, S, V0, k, P0, Ta = (X[:, j] for j in range(6))
    T0 = _PIS_VALS[level - 1]
    A = P0 * S + 19.62 * M - k * V0 / S
    V = (S / (2.0 * k)) * (np.sqrt(A ** 2 + 4.0 * k * P0 * V0 * Ta / T0) - A)
    return 2.0 * np.pi * np.sqrt(M / (k + S ** 2 * P0 * V0 * Ta / (T0 * V ** 2)))


def _piston_sigma(X, level):
    M, P0 = X[:, 0], X[:, 4]
    return (0.002 * (1.0 + 3.0 * (P0 - 90000.0) / 20000.0)
            * (1.0 + 0.5 * (M - 30.0) / 30.0) * _PIS_MULS[level - 1])


# Ground truth: unique boundary optimum, same corner for every level (multi-start verified, zero
# spread). NOTE the per-level f* gaps (~1e-3) are far below sigma (up to ~0.03): identifying the best
# LEVEL under noise is the intended challenge of this problem.
_PIS_XSTAR = [30.0, 0.020, 0.002, 5000.0, 110000.0, 290.0]
_PIS_FSTAR = [0.1674451638202761, 0.16645996229564428, 0.1654924039747633, 0.1645419534505813]
PISTON = ProblemSpec(
    "piston", _piston_f, _piston_sigma, n_levels=4,
    bounds=[(30.0, 60.0), (0.005, 0.020), (0.002, 0.010), (1000.0, 5000.0),
            (90000.0, 110000.0), (290.0, 296.0)],
    meta=dict(cat_values=_PIS_VALS.tolist(), noise_muls=_PIS_MULS.tolist(),
              f_star=_PIS_FSTAR[3], opt_level=4, x_star=_PIS_XSTAR, f_star_per_level=_PIS_FSTAR),
)


# ======================================================================================
#  ENG-3  OTL circuit -- 5 continuous [Rb1, Rb2, Rf, Rc1, Rc2] + beta as the 4-level categorical
#  {50,100,200,300}. Mid-point voltage:
#     Vm = 12 Rb2 / (Rb1 + Rb2);  den = beta (Rc2 + 9) + Rf
#     f = (Vm + 0.74) beta (Rc2 + 9) / den + 11.35 Rf / den + 0.74 Rf beta (Rc2 + 9) / (den Rc1)
#     sigma = 0.01 sqrt(Rb1 / 50) (1 + 0.3 Rf) * mult(l)
# ======================================================================================
_OTL_VALS = np.array([50.0, 100.0, 200.0, 300.0])
_OTL_MULS = np.array([2.5, 1.5, 1.0, 0.7])


def _otl_f(X, level):
    Rb1, Rb2, Rf, Rc1, Rc2 = (X[:, j] for j in range(5))
    beta = _OTL_VALS[level - 1]
    Vm = 12.0 * Rb2 / (Rb1 + Rb2)
    bR = beta * (Rc2 + 9.0)
    den = bR + Rf
    return (Vm + 0.74) * bR / den + 11.35 * Rf / den + 0.74 * Rf * bR / (den * Rc1)


def _otl_sigma(X, level):
    return 0.01 * np.sqrt(X[:, 0] / 50.0) * (1.0 + 0.3 * X[:, 2]) * _OTL_MULS[level - 1]


# Ground truth: unique boundary optimum, same corner for every level (multi-start verified, zero
# spread). Per-level f* gaps (~2e-3..7e-3) sit below sigma (up to ~0.1) -> level identification under
# noise is the challenge here too.
_OTL_XSTAR = [150.0, 25.0, 0.5, 2.5, 1.25]
_OTL_FSTAR = [2.610811751601225, 2.6065508114508598, 2.6044187828752565, 2.6037078756067538]
OTL = ProblemSpec(
    "otl_circuit", _otl_f, _otl_sigma, n_levels=4,
    bounds=[(50.0, 150.0), (25.0, 70.0), (0.5, 3.0), (1.2, 2.5), (0.25, 1.25)],
    meta=dict(cat_values=_OTL_VALS.tolist(), noise_muls=_OTL_MULS.tolist(),
              f_star=_OTL_FSTAR[3], opt_level=4, x_star=_OTL_XSTAR, f_star_per_level=_OTL_FSTAR),
)


# ======================================================================================
#  Phase-2 multi-dim problems (TP-5/6, ENG-1/2/3) -- stubs until their equations are added.
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
    "griewank_10d":  GRIEWANK10D,             # TP-5  (9-D, 4 levels) LIVE
    "ackley_10d":    ACKLEY10D,                # TP-6  (9-D, 4 levels) LIVE
    "rastrigin_6d":  RASTRIGIN6D,              # TP-7  (5-D, 4 levels) LIVE
    "golinski":      GOLINSKI,                 # ENG-1 (6-D, 5 levels) LIVE
    "piston":        PISTON,                   # ENG-2 (6-D, 4 levels) LIVE
    "otl_circuit":   OTL,                      # ENG-3 (5-D, 4 levels) LIVE
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
            sp.f_true_level(sp.bounds.mean(axis=1), 1)   # mid-box probe (zeros can be out of bounds)
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
    if "f_star" in spec.meta:                        # analytic optimum (multi-dim problems)
        return float(spec.meta["f_star"])
    _grid_1d_only(spec, "ground_truth_min")
    x1 = np.linspace(spec.lb, spec.ub, n)
    return float(min(spec.f_true_level(x1, lv).min() for lv in spec.levels))


def true_opt_location(spec, n=4000):
    if "x_star" in spec.meta:
        return int(spec.meta["opt_level"]), np.asarray(spec.meta["x_star"], float)
    _grid_1d_only(spec, "true_opt_location")
    x1 = np.linspace(spec.lb, spec.ub, n)
    best = (np.inf, None, None)
    for lv in spec.levels:
        fv = spec.f_true_level(x1, lv)
        if fv.min() < best[0]:
            best = (fv.min(), lv, x1[fv.argmin()])
    return best[1], best[2]


def true_min_per_category(spec, n=4000):
    if "f_star_per_level" in spec.meta:
        return np.asarray(spec.meta["f_star_per_level"], float)
    _grid_1d_only(spec, "true_min_per_category")
    x1 = np.linspace(spec.lb, spec.ub, n)
    return np.array([spec.f_true_level(x1, lv).min() for lv in spec.levels])
