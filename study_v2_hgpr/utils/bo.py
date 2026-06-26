"""
bo.py -- BO loop for the per-category HGPR study (study_v2_hgpr).

Structure mirrors study_v2_gp/bo.py (enumerate categories, optimize the 1-D acquisition per
category with a dense grid + L-BFGS, pick the global best, one shared incumbent), with two
differences driven by the HGPR surrogate:
  * the per-category data keeps the RAW REPLICATES (HGPR learns sigma^2(x) from the scatter), and
  * the aleatoric r(x) for haei/anpei/rahbo comes from the HGPR itself (model.r), not a separate poly.

Cost control: a cold HGPR fit is ~2000 Adam epochs, but each iteration changes only one category,
so we cold-fit all 5 once and then WARM-refit just the changed category (model.refit_level).
Saved .npz schema is identical to study_driver.m so results.py / the comparison overlays work.
"""
import time
import numpy as np
from scipy.optimize import minimize as _minimize

from . import problem, acquisitions
from .model import PerCategoryHGPR

GRID_N = 256           # dense seed grid for the 1-D continuous search per category
N_TOP = 3              # L-BFGS-B restarts from the best grid seeds


def _minimize_1d(g, lb, ub):
    """Minimize scalar g(x) over [lb, ub]: dense grid seed + L-BFGS-B polish. g is vectorized."""
    xs = np.linspace(lb, ub, GRID_N)
    vals = np.asarray(g(xs), float)
    order = np.argsort(vals)
    best_x, best_v = float(xs[order[0]]), float(vals[order[0]])
    g1 = lambda z: float(g(np.array([z[0]]))[0])
    for idx in order[:N_TOP]:
        res = _minimize(g1, x0=[float(xs[idx])], bounds=[(lb, ub)], method="L-BFGS-B")
        if res.success and float(res.fun) < best_v:
            best_v, best_x = float(res.fun), float(res.x[0])
    return best_x, best_v


def _argbest_over_categories(per_level_fn, levels, lb, ub):
    best = (None, None, np.inf)
    for lv in levels:
        x, v = _minimize_1d(per_level_fn(lv), lb, ub)
        if v < best[2]:
            best = (lv, x, v)
    return best


def run_bo(acf, param, n_rep, seed, num_iter):
    """Run one BO cell. Returns a dict of numpy arrays + meta, ready to np.savez."""
    t0 = time.time()
    lb, ub = problem.LB, problem.UB
    rng = np.random.default_rng(seed)

    doe = problem.initial_doe(n_rep, rng=rng)
    Xs = doe["X_sample"]; Ys = doe["Y_sample"].copy(); Vs = doe["Var_sample"].copy()
    Yrep = doe["Y_rep"]
    n_initial = Xs.shape[0]

    # per-category running data, keeping RAW replicates
    data = {}
    for lv in problem.LEVELS:
        idx = np.where(Xs[:, 1].astype(int) == lv)[0]
        data[int(lv)] = dict(x1=[float(Xs[i, 0]) for i in idx],
                             y_rep=[np.asarray(Yrep[i], float) for i in idx],
                             y_mean=[float(Ys[i]) for i in idx],
                             y_var=[float(Vs[i]) for i in idx])

    X_sampled = list(map(list, Xs)); Y_sampled = list(Ys); Y_var_sampled = list(Vs)
    Y_min_history, Y_min_est, X_min_est = [], [], []
    X_next_history, Y_next_history, Y_var_next_history = [], [], []
    needs_r = acquisitions.needs_aleatoric(acf)

    gps = PerCategoryHGPR.cold_fit(data)                      # cold-fit all 5 once

    for _ in range(num_iter):
        ymin = float(np.min(Y_sampled))                      # global incumbent (best sample-mean)

        # recommended optimum: argmin posterior mean across categories
        mean_fn = lambda lv: (lambda xs: gps.predict(lv, xs, observation_noise=False)[0])
        lv_est, x_est, y_est = _argbest_over_categories(mean_fn, gps.levels, lb, ub)
        X_min_est.append([x_est, lv_est]); Y_min_est.append(y_est)

        # next point: argmin acquisition U_negate across categories (shared ymin)
        def acq_fn(lv):
            def g(xs):
                mu, s = gps.mean_std(lv, xs)
                r = gps.r(lv, xs) if needs_r else None
                return acquisitions.evaluate(acf, mu, s, ymin, r=r, param=param)
            return g
        lv_next, x_next, _ = _argbest_over_categories(acq_fn, gps.levels, lb, ub)

        # evaluate noisy objective n_rep x; append RAW replicates to that category
        y_rep = problem.noisy_eval(x_next, lv_next, n_rep, rng)
        y_mean, y_var = float(y_rep.mean()), float(y_rep.var(ddof=1))
        d = data[lv_next]
        d["x1"].append(float(x_next)); d["y_rep"].append(np.asarray(y_rep, float))
        d["y_mean"].append(y_mean); d["y_var"].append(y_var)
        X_sampled.append([x_next, lv_next]); Y_sampled.append(y_mean); Y_var_sampled.append(y_var)
        X_next_history.append([x_next, lv_next]); Y_next_history.append(y_mean)
        Y_var_next_history.append(y_var)
        Y_min_history.append(float(np.min(Y_sampled)))

        gps.refit_level(lv_next, d)                          # warm-refit only the changed category

    Y_sampled_arr = np.asarray(Y_sampled)
    bi = int(np.argmin(Y_sampled_arr))
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
        X_best_final=np.asarray(X_sampled[bi], float),
        Y_best_final=np.asarray([float(Y_sampled_arr[bi])]),
        Y_var_best_final=np.asarray([float(Y_var_sampled[bi])]),
        n_initial=np.asarray([n_initial]),
        var_fctr=problem.VAR_FCTR,
        meta=dict(acf=acf, acf_param=float(param), n_rep=int(n_rep), seed=int(seed),
                  num_iter=int(num_iter), runtime=float(runtime), model="hgpr"),
    )
