"""
bo.py -- model-AGNOSTIC Python BO loop (generalizes study_v2_gp/utils/bo.py).

Given a ProblemSpec and a python Model (SeparateGP / CategoricalKernelGP / any future one that
implements fit/mean_std/r), each iteration:
  1. fit the model on the current per-category data (+ aleatoric if the acq needs r)
  2. recommended optimum = argmin posterior mean over categories
  3. next point = argmin acquisition U_negate over categories, using ONE global incumbent
  4. evaluate the noisy objective n_rep x at the chosen (x1, level), append
Saves the study_driver.m field schema so results.py / the comparison layer work for every model.
"""
import time
import numpy as np
from scipy.optimize import minimize as _minimize

from . import problems as P
from . import acquisitions

GRID_N = 256
N_TOP = 3


def _minimize_1d(g, lb, ub):
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


def run_bo(spec, model_cls, acf, param, n_rep, seed, num_iter, model_name="python"):
    """One BO cell. spec = ProblemSpec; model_cls = python Model class. Returns dict for np.savez."""
    t0 = time.time()
    lb, ub = spec.lb, spec.ub
    rng = np.random.default_rng(seed)

    doe = P.initial_doe(spec, n_rep, rng=rng)
    Xs = doe["X_sample"]; Ys = doe["Y_sample"].copy(); Vs = doe["Var_sample"].copy()
    n_initial = Xs.shape[0]

    data = {}
    for lv in spec.levels:
        m = Xs[:, 1].astype(int) == lv
        data[int(lv)] = dict(x1=list(Xs[m, 0]), y_mean=list(Ys[m]), y_var=list(Vs[m]))

    X_sampled = list(map(list, Xs)); Y_sampled = list(Ys); Y_var_sampled = list(Vs)
    Y_min_history, Y_min_est, X_min_est = [], [], []
    X_next_history, Y_next_history, Y_var_next_history = [], [], []
    needs_r = acquisitions.needs_aleatoric(acf)

    for _ in range(num_iter):
        model = model_cls.fit(data, needs_r=needs_r)
        ymin = float(np.min(Y_sampled))

        mean_fn = lambda lv: (lambda xs: model.predict(lv, xs, observation_noise=False)[0])
        lv_est, x_est, y_est = _argbest_over_categories(mean_fn, model.levels, lb, ub)
        X_min_est.append([x_est, lv_est]); Y_min_est.append(y_est)

        def acq_fn(lv):
            def g(xs):
                mu, s = model.mean_std(lv, xs)
                r = model.r(lv, xs) if needs_r else None
                return acquisitions.evaluate(acf, mu, s, ymin, r=r, param=param)
            return g
        lv_next, x_next, _ = _argbest_over_categories(acq_fn, model.levels, lb, ub)

        y_rep = P.noisy_eval(spec, x_next, lv_next, n_rep, rng)
        y_mean, y_var = float(y_rep.mean()), float(y_rep.var(ddof=1))
        data[lv_next]["x1"].append(x_next); data[lv_next]["y_mean"].append(y_mean)
        data[lv_next]["y_var"].append(y_var)
        X_sampled.append([x_next, lv_next]); Y_sampled.append(y_mean); Y_var_sampled.append(y_var)
        X_next_history.append([x_next, lv_next]); Y_next_history.append(y_mean)
        Y_var_next_history.append(y_var)
        Y_min_history.append(float(np.min(Y_sampled)))

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
        meta=dict(problem=spec.name, model=model_name, acf=acf, acf_param=float(param),
                  n_rep=int(n_rep), seed=int(seed), num_iter=int(num_iter), runtime=float(runtime)),
    )
