"""
bo.py -- the Bayesian-optimization loop for the per-category-GP baseline.

Mirrors Heter_BO_GF/find_next.m, but with INDEPENDENT GPs per category instead of one
LVGP. The categorical choice is handled by ENUMERATION (no GP models it):

    each iteration:
      1. fit one GP + one aleatoric poly per category on that category's data
      2. recommended optimum X_min_est = argmin over categories of (argmin_x1 posterior mean)
      3. next point  X_next       = argmin over categories of (argmin_x1 acquisition U_negate)
         -- every category's acquisition uses the SAME global incumbent
            ymin = min observed sample-mean, so the values are comparable across categories
      4. evaluate the noisy objective n_rep x at X_next, append to that category's data

Output dict uses the SAME field names as study_v2/study_driver.m so the analysis layer
(utils/results.py) and the true-regret metric work for both studies.
"""
import time
import numpy as np
from scipy.optimize import minimize as _minimize

from . import problem, acquisitions
from .model import PerCategoryGPs
from .aleatoric import AleatoricModels

GRID_N = 256           # dense seed grid for the 1-D continuous search per category
N_TOP = 3              # L-BFGS-B restarts from the best grid seeds (cf. find_next's n_top)

# knob name per family (for reference / labels); the value is the swept `param`
KNOB = {"haei": "gamma", "anpei": "beta_anpei", "rahbo": "alpha"}


def _minimize_1d(g, lb, ub):
    """Minimize scalar g(x) over [lb, ub]: dense grid seed + L-BFGS-B polish on top seeds.
    g must be vectorized (accept a 1-D array, return a 1-D array). Returns (x*, g*)."""
    xs = np.linspace(lb, ub, GRID_N)
    vals = np.asarray(g(xs), float)
    order = np.argsort(vals)
    best_x, best_v = float(xs[order[0]]), float(vals[order[0]])
    g1 = lambda z: float(g(np.array([z[0]]))[0])             # scalar wrapper for the optimizer
    for idx in order[:N_TOP]:
        res = _minimize(g1, x0=[float(xs[idx])], bounds=[(lb, ub)], method="L-BFGS-B")
        if res.success and float(res.fun) < best_v:
            best_v, best_x = float(res.fun), float(res.x[0])
    return best_x, best_v


def _argbest_over_categories(per_level_fn, levels, lb, ub):
    """Run _minimize_1d for each level; return (best_level, best_x, best_val) globally."""
    best = (None, None, np.inf)
    for lv in levels:
        x, v = _minimize_1d(per_level_fn(lv), lb, ub)
        if v < best[2]:
            best = (lv, x, v)
    return best


def run_bo(acf, param, n_rep, seed, num_iter):
    """Run one BO cell. Returns a dict of numpy arrays + meta, ready to np.savez.

    acf    : 'ei'|'lcb'|'pi'|'haei'|'anpei'|'rahbo'
    param  : family knob (gamma/beta_anpei/alpha); NaN for ei/lcb/pi
    n_rep  : replicates per evaluated location
    seed   : controls DOE + all noise (one rng stream)
    num_iter: number of BO iterations
    """
    t0 = time.time()
    lb, ub = problem.LB, problem.UB
    rng = np.random.default_rng(seed)                        # ONE stream: DOE + in-loop noise

    # ---- initial DOE (shared maximin LHS, n_rep replicates per (loc, level)) ----
    doe = problem.initial_doe(n_rep, rng=rng)
    Xs = doe["X_sample"]                                     # (10, 2) [x1, level]
    Ys = doe["Y_sample"].copy()                             # (10,)
    Vs = doe["Var_sample"].copy()                           # (10,)
    n_initial = Xs.shape[0]

    # per-category running data {level: dict(x1, y_mean, y_var)}
    data = {}
    for lv in problem.LEVELS:
        m = Xs[:, 1].astype(int) == lv
        data[int(lv)] = dict(x1=list(Xs[m, 0]), y_mean=list(Ys[m]), y_var=list(Vs[m]))

    X_sampled = list(map(list, Xs))                          # grows with BO points
    Y_sampled = list(Ys)
    Y_var_sampled = list(Vs)

    Y_min_history, Y_min_est, X_min_est = [], [], []
    X_next_history, Y_next_history, Y_var_next_history = [], [], []
    needs_r = acquisitions.needs_aleatoric(acf)

    for _ in range(num_iter):
        gps = PerCategoryGPs.fit(data)
        ale = AleatoricModels.fit(data) if needs_r else None
        ymin = float(np.min(Y_sampled))                      # global incumbent (best sample-mean)

        # 2) recommended optimum: argmin posterior mean across categories
        mean_fn = lambda lv: (lambda xs: gps.predict(lv, xs, observation_noise=False)[0])
        lv_est, x_est, y_est = _argbest_over_categories(mean_fn, gps.levels, lb, ub)
        X_min_est.append([x_est, lv_est]); Y_min_est.append(y_est)

        # 3) next point: argmin acquisition U_negate across categories (shared ymin)
        def acq_fn(lv):
            def g(xs):
                mu, s = gps.mean_std(lv, xs)
                r = ale.r(lv, xs) if needs_r else None
                return acquisitions.evaluate(acf, mu, s, ymin, r=r, param=param)
            return g
        lv_next, x_next, _ = _argbest_over_categories(acq_fn, gps.levels, lb, ub)

        # 4) evaluate noisy objective n_rep x at the chosen (x1, level)
        y_rep = problem.noisy_eval(x_next, lv_next, n_rep, rng)
        y_mean, y_var = float(y_rep.mean()), float(y_rep.var(ddof=1))
        data[lv_next]["x1"].append(x_next)
        data[lv_next]["y_mean"].append(y_mean)
        data[lv_next]["y_var"].append(y_var)
        X_sampled.append([x_next, lv_next]); Y_sampled.append(y_mean); Y_var_sampled.append(y_var)
        X_next_history.append([x_next, lv_next]); Y_next_history.append(y_mean)
        Y_var_next_history.append(y_var)
        Y_min_history.append(float(np.min(Y_sampled)))       # best observed sample-mean so far

    # ---- final best OBSERVED design ----
    Y_sampled_arr = np.asarray(Y_sampled)
    bi = int(np.argmin(Y_sampled_arr))
    X_best_final = np.asarray(X_sampled[bi], float)
    Y_best_final = float(Y_sampled_arr[bi])
    Y_var_best_final = float(Y_var_sampled[bi])
    runtime = time.time() - t0

    return dict(
        Y_min_history=np.asarray(Y_min_history),
        X_sampled=np.asarray(X_sampled, float),
        Y_sampled=Y_sampled_arr,
        Y_var_sampled=np.asarray(Y_var_sampled),
        X_next_history=np.asarray(X_next_history, float),
        Y_next_history=np.asarray(Y_next_history),
        Y_var_next_history=np.asarray(Y_var_next_history),
        Y_min_est=np.asarray(Y_min_est),
        X_min_est=np.asarray(X_min_est, float),
        X_best_final=X_best_final,
        Y_best_final=np.asarray([Y_best_final]),
        Y_var_best_final=np.asarray([Y_var_best_final]),
        n_initial=np.asarray([n_initial]),
        var_fctr=problem.VAR_FCTR,
        meta=dict(acf=acf, acf_param=float(param), n_rep=int(n_rep), seed=int(seed),
                  num_iter=int(num_iter), runtime=float(runtime), model="percat_gp"),
    )
