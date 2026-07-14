"""
doe.py -- initial-design generators over the mixed (continuous x1 + categorical level) space.

Each generator returns {level(1-based): sorted array of x1 locations}. Modes:
  "shared"    : n_tr_lv maximin-LHS x1 on a 1/6 inset, the SAME at every level (current default;
                two tight vertical bands -- under-explores x1 per category).
  "per_level" : each level an INDEPENDENT maximin-LHS in x1 over [lb,ub] (spread within a level, but
                levels are uncoordinated -- can overlap in x1).
  "slhd"      : Sliced Latin Hypercube Design -- each level is an LHS in x1 AND the union across
                levels is a full LHS in x1 (balanced + space-filling; recommended for LVGP).
  "joint"     : one 2-D LHS over (x1, category); ~n_tr_lv points per level at varied x1.

`n_tr_lv` = points per level (total = n_lv * n_tr_lv). All map [0,1] -> the chosen x-range.
"""
from typing import NamedTuple

import numpy as np

EDGE_BUF = 1.0 / 6.0


def _maximin_lhs_1d(rng, n, n_iter=8000):
    """maximin LHS of n points in [0,1] (emulates MATLAB lhsdesign default)."""
    edges = np.linspace(0.0, 1.0, n + 1)
    lo_e, hi_e = edges[:-1], edges[1:]
    cand = rng.uniform(lo_e, hi_e, size=(n_iter, n))
    cand.sort(axis=1)
    gaps = np.diff(cand, axis=1).min(axis=1) if n > 1 else np.ones(n_iter)
    return cand[int(gaps.argmax())]


def _lhs_1d(rng, n):
    """plain LHS of n points in [0,1] (one jittered point per stratum, random order)."""
    return np.sort((np.arange(n) + rng.uniform(0, 1, n)) / n)


def _scale(u, lb, ub):
    return lb + np.asarray(u) * (ub - lb)


def _chunk_sizes(N, n_lv, rng):
    """Split N design points into consecutive stratum-chunks: `q = N//n_lv` full chunks of n_lv, plus
    (if N is not a multiple of n_lv) ONE short chunk of r = N % n_lv. The short chunk's POSITION is
    randomized -- otherwise the r extra points would always land in the top r/N of the domain."""
    q, r = divmod(N, n_lv)
    sizes = [n_lv] * q + ([r] if r else [])
    rng.shuffle(sizes)
    return sizes


def extra_levels(N, n_lv, rng):
    """The r = N % n_lv distinct levels (0-based) that receive one extra design point. Chosen ONCE and
    reused across continuous dims by make_doe_nd, so every dim agrees on each level's point count."""
    r = N % n_lv
    return np.sort(rng.permutation(n_lv)[:r])


def make_doe(mode, rng, lb, ub, n_lv, n_tr_lv=None, inset=EDGE_BUF, n_init=None, extra=None):
    """Return {level: x1 array} for the given DOE `mode`.

    Size: give EITHER n_tr_lv (points per level, total = n_lv*n_tr_lv) or n_init (total points).
    n_init need not divide n_lv -- `slhd`/`per_level`/`joint` handle the remainder by giving r = n_init
    % n_lv levels one extra point. Pass `extra` (from extra_levels()) to fix WHICH levels those are.
    (`shared` repeats one design at every level, so it requires divisibility by construction.)"""
    if (n_tr_lv is None) == (n_init is None):
        raise ValueError("give exactly one of n_tr_lv / n_init")
    N = n_lv * n_tr_lv if n_init is None else int(n_init)
    q, r = divmod(N, n_lv)
    if r and extra is None:
        extra = extra_levels(N, n_lv, rng)
    extra = np.asarray([] if extra is None else extra, int)

    if mode == "shared":
        if r:
            raise ValueError(f"mode 'shared' needs n_init ({N}) divisible by n_lv ({n_lv})")
        lo, hi = lb + inset * (ub - lb), ub - inset * (ub - lb)
        x = _scale(_maximin_lhs_1d(rng, q), lo, hi)
        return {lv: x.copy() for lv in range(1, n_lv + 1)}

    if mode == "per_level":
        return {lv: np.sort(_scale(_maximin_lhs_1d(rng, q + int((lv - 1) in extra)), lb, ub))
                for lv in range(1, n_lv + 1)}

    if mode == "slhd":
        # True 1-D sliced LHD (Qian 2012). Walk the N equal strata in consecutive chunks of n_lv; in
        # each chunk a fresh permutation hands every level exactly ONE stratum. So each level gets one
        # point per chunk -> each level is an LHS spanning the range, AND the union of all N points is
        # a full LHS (every stratum used exactly once). That per-level guarantee is what `joint` lacks.
        # A short chunk (n_init not a multiple of n_lv) gives the `extra` levels one more point.
        out = {lv: [] for lv in range(1, n_lv + 1)}
        s = 0
        for size in _chunk_sizes(N, n_lv, rng):
            lvs = rng.permutation(n_lv) if size == n_lv else rng.permutation(extra)
            for k, lv in enumerate(lvs):
                u = (s + k + rng.uniform(0, 1)) / N     # stratum s+k, jittered inside it
                out[lv + 1].append(_scale(u, lb, ub))
            s += size
        return {lv: np.sort(np.array(v)) for lv, v in out.items()}

    if mode == "joint":
        # One LHS over (x1, category): x1 strata shuffled against a balanced set of category labels.
        # When n_lv divides N the sorted-category-strata construction yields exactly q per level, so
        # building the labels explicitly is equivalent -- and it keeps the level COUNTS fixed when it
        # does not divide (otherwise each continuous dim would round to a different count).
        ux = _lhs_1d(rng, N)[rng.permutation(N)]        # x1 strata, shuffled
        labels = np.repeat(np.arange(n_lv), q)
        if r:
            labels = np.concatenate([labels, extra])
        labels = labels[rng.permutation(N)]
        out = {lv: [] for lv in range(1, n_lv + 1)}
        for xu, lv in zip(ux, labels):
            out[lv + 1].append(_scale(xu, lb, ub))
        return {lv: np.sort(np.array(v)) for lv, v in out.items()}

    raise ValueError(f"unknown DOE mode '{mode}' (shared|per_level|slhd|joint)")


MODES = ["shared", "per_level", "slhd", "joint"]


# ======================================================================================
#  d-dimensional DOE + the full test-suite structure (for the DOE decision across all problems)
# ======================================================================================
def make_doe_nd(mode, rng, bounds, n_lv, n_tr_lv=None, n_init=None):
    """d-dim DOE over `d` continuous dims x `n_lv` categorical levels.
    bounds = (d, 2) array-like of [lb, ub] per dim. Returns {level: (m_lv, d) array}.
    Each continuous dim gets the chosen 1-D design; within a level the dims are shuffled
    independently to decorrelate them (a proper d-dim LHS per slice).

    With a remainder (n_init not a multiple of n_lv) the levels receiving the extra point are drawn
    ONCE and shared by every dim -- otherwise the per-dim columns would have mismatched lengths.

    d == 1 REDUCES EXACTLY to make_doe: the per-dim shuffle (which only decorrelates dims against each
    other, and is a no-op on a single column) is SKIPPED so it consumes no RNG. That keeps the 1-D DOE
    -- and every noise draw that follows it from the same rng -- bit-identical to the pre-d-dim code."""
    bounds = np.atleast_2d(np.asarray(bounds, float))
    d = bounds.shape[0]
    N = n_lv * n_tr_lv if n_init is None else int(n_init)
    ex = extra_levels(N, n_lv, rng) if N % n_lv else None
    per_dim = [make_doe(mode, rng, bounds[j, 0], bounds[j, 1], n_lv,
                        n_tr_lv=n_tr_lv, n_init=n_init, extra=ex) for j in range(d)]
    out = {}
    for lv in range(1, n_lv + 1):
        cols = []
        for j in range(d):
            v = np.asarray(per_dim[j][lv], float).copy()
            if d > 1:
                rng.shuffle(v)                      # decorrelate dims within the slice (no-op at d=1)
            cols.append(v)
        out[lv] = np.column_stack(cols)             # (m_lv, d)
    return out


METRICS = ["pooled_uniformity", "per_level_cov", "pooled_maximin", "gower_maximin"]


def doe_metrics_nd(mode, bounds, n_lv, n_tr_lv=None, n_seeds=200, n_init=None):
    """Consolidated space-filling metrics for a d-dim DOE (all higher = better), averaged over seeds.
    Same four metrics are used for 1-D and d-dim (call with bounds=[(lb,ub)] for 1-D).

    PRIMARY -- these decide the verdict:
      pooled_uniformity : per-dim min consecutive gap of the POOLED (all-level) values x N, avg over
                          dims (1.0 = perfect LHS spacing). Discrepancy-type; SLHD's union-is-LHS
                          guarantee. shared=0 (duplicates), per_level~=0 (uncoordinated union).
      per_level_cov %   : each level covers all n_tr_lv bins in EVERY dim (per-category coverage).
                          SLHD=100% by construction; joint drops (random assignment leaves gaps).
    SECONDARY -- honesty / distance criteria that FAVOR per_level (which optimizes distance); SLHD
    trades a little of these for the stratification the shared surrogates need:
      pooled_maximin    : min pairwise Euclidean distance among all continuous points (level ignored).
      gower_maximin     : min pairwise MIXED Gower distance = (sum_j |dx_j|_norm + 1{different level})
                          / (d+1) -- continuous L1 + a Hamming term for the NOMINAL level. Dominated by
                          same-level pairs, so it rewards within-level separation and is blind to
                          cross-level coordination (why a single mixed distance can't rank the modes)."""
    from scipy.spatial.distance import pdist
    bounds = np.atleast_2d(np.asarray(bounds, float))
    d = bounds.shape[0]
    span = np.where(bounds[:, 1] > bounds[:, 0], bounds[:, 1] - bounds[:, 0], 1.0)
    N = n_lv * n_tr_lv if n_init is None else int(n_init)
    nb = N // n_lv                                   # coverage bins per dim (floor share of a level)
    if mode == "shared" and N % n_lv:
        # `shared` repeats ONE design at every level, so it cannot express a non-divisible n_init.
        return dict(pooled_uniformity=np.nan, per_level_cov=np.nan,
                    pooled_maximin=np.nan, gower_maximin=np.nan)
    unif, pcov, pm, gow = [], [], [], []
    for s in range(n_seeds):
        dd = make_doe_nd(mode, np.random.default_rng(s), bounds, n_lv,
                         n_tr_lv=n_tr_lv, n_init=n_init)
        norm = {lv: (v - bounds[:, 0]) / span for lv, v in dd.items()}
        allp = np.vstack(list(norm.values()))
        lvs = np.concatenate([[lv] * len(norm[lv]) for lv in norm])
        us = []
        for j in range(d):
            col = np.sort(allp[:, j])
            us.append(np.min(np.diff(col)) * len(col) if len(col) > 1 else 0.0)
        unif.append(np.mean(us))
        pcov.append(np.mean([len(set(np.clip((norm[lv][:, j] * nb).astype(int), 0, nb - 1))) == nb
                             for lv in norm for j in range(d)]))
        if len(allp) > 1:
            pm.append(float(pdist(allp).min()))
            cont = pdist(allp, "cityblock")                       # sum_j |dx_j|_norm per pair
            iu = np.triu_indices(len(lvs), k=1)
            differ = (lvs[:, None] != lvs[None, :])[iu].astype(float)
            gow.append(float(((cont + differ) / (d + 1)).min()))
        else:
            pm.append(0.0); gow.append(0.0)
    return dict(pooled_uniformity=np.mean(unif), per_level_cov=100 * np.mean(pcov),
                pooled_maximin=np.mean(pm), gower_maximin=np.mean(gow))


class SuiteSpec(NamedTuple):
    """Geometry + budget of one test problem. SINGLE SOURCE OF TRUTH for both.
    n_init_xlsx / num_iter come from resources/init_doe_iter.xlsx; d/bounds/n_levels from the equations PDF.
    The DOE needs only dims/bounds/levels -- not the equations -- so this covers the stubbed problems too.

    A proper sliced LHD needs n_levels | n_init: only then is every slice an LHS on equal-width bins.
    Five problems' spreadsheet sizes are not (50/4, 30/4), so `n_init` ROUNDS UP to the next multiple of
    n_levels -- never spending less than the spreadsheet asks. Verified: rounding restores per-level
    coverage to 100% (at 50 it is 19%). See notebooks/doe.ipynb."""
    d: int                  # continuous dims
    bounds: list            # [(lb,ub)] per continuous dim
    n_levels: int           # categorical levels
    n_init_xlsx: int        # TOTAL initial design points as specified in the spreadsheet
    num_iter: int           # BO iterations

    @property
    def n_init(self):       # spreadsheet size rounded UP to a multiple of n_levels (SLHD requirement)
        return -(-self.n_init_xlsx // self.n_levels) * self.n_levels

    @property
    def n_tr_lv(self):      # design points per level (exact -- n_init is divisible by construction)
        return self.n_init // self.n_levels


PROBLEM_GRID = {
    #  n_init below is the RAW spreadsheet value; .n_init rounds it up to a multiple of n_levels
    #  (50->52, 30->32 for the four-level problems; the rest already divide exactly).
    #                    d  bounds                                                              lv  n_init iter
    "branin_hetero": SuiteSpec(1, [(-5, 10)],                                                    5,  10,  50),
    "sixhump_camel": SuiteSpec(1, [(-2, 2)],                                                     4,   8,  50),
    "griewank_2d":   SuiteSpec(1, [(-5, 5)],                                                     4,   8,  50),
    "ackley_2d":     SuiteSpec(1, [(-3, 3)],                                                     4,   8,  50),
    "griewank_10d":  SuiteSpec(9, [(-5, 5)] * 9,                                                 4,  50, 200),
    "ackley_10d":    SuiteSpec(9, [(-5, 5)] * 9,                                                 4,  50, 200),
    "rastrigin_6d":  SuiteSpec(5, [(-5, 5)] * 5,                                                 4,  30, 200),
    "golinski":      SuiteSpec(6, [(2.6, 3.6), (0.7, 0.8), (7.3, 8.3), (7.3, 8.3), (2.9, 3.9), (5.0, 5.5)],
                                                                                                 5,  30, 200),
    "piston":        SuiteSpec(6, [(30, 60), (0.005, 0.020), (0.002, 0.010), (1000, 5000),
                                   (90000, 110000), (290, 296)],                                 4,  30, 200),
    "otl_circuit":   SuiteSpec(5, [(50, 150), (25, 70), (0.5, 3.0), (1.2, 2.5), (0.25, 1.25)],   4,  30, 200),
}
