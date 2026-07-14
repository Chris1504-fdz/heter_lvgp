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
from . import doe_cache

GRID_N = 256
N_TOP = 3
SOBOL_MIN = 256                # d>1: candidates = next power of 2 >= max(SOBOL_MIN, SOBOL_PER_DIM*d)
SOBOL_PER_DIM = 128


def _minimize_1d(g, lb, ub):
    """The ORIGINAL 1-D optimizer (dense grid + L-BFGS polish from the top 3). g takes a (n,) array.
    Kept verbatim: at d=1 a 256-point grid is effectively exhaustive, and keeping it byte-identical
    preserves the bit-exact regression against the stored 1-D results."""
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


def _minimize_box(g, bounds):
    """Minimize g over a d-dim box. g takes an (n, d) array, returns (n,).
    d == 1 -> the exact legacy path above (bit-exact with the stored 1-D results).
    d > 1  -> deterministic unscrambled-Sobol candidates + L-BFGS-B polish from the top max(3, d):
              the same architecture as 1-D (global scan then local polish), scaled by dimension.
              Deterministic on purpose, so re-running a cell reproduces it exactly."""
    bounds = np.atleast_2d(np.asarray(bounds, float))
    d = bounds.shape[0]
    if d == 1:
        x, v = _minimize_1d(lambda xs: g(np.asarray(xs, float).reshape(-1, 1)),
                            bounds[0, 0], bounds[0, 1])
        return np.array([x]), v
    from scipy.stats import qmc
    m = int(np.ceil(np.log2(max(SOBOL_MIN, SOBOL_PER_DIM * d))))
    u = qmc.Sobol(d, scramble=False).random(2 ** m)
    cand = bounds[:, 0] + u * (bounds[:, 1] - bounds[:, 0])
    vals = np.asarray(g(cand), float)
    order = np.argsort(vals)
    best_x, best_v = cand[order[0]].copy(), float(vals[order[0]])
    g1 = lambda z: float(g(np.asarray(z, float).reshape(1, -1))[0])
    for idx in order[:max(N_TOP, d)]:
        res = _minimize(g1, x0=cand[idx], bounds=[tuple(b) for b in bounds], method="L-BFGS-B")
        if res.success and float(res.fun) < best_v:
            best_v, best_x = float(res.fun), np.asarray(res.x, float)
    return best_x, best_v


def _argbest_over_categories(per_level_fn, levels, bounds):
    """Best (level, x, value) over all categories; x is a (d,) array."""
    best = (None, None, np.inf)
    for lv in levels:
        x, v = _minimize_box(per_level_fn(lv), bounds)
        if v < best[2]:
            best = (lv, x, v)
    return best


def run_bo(spec, model_cls, acf, param, n_rep, seed, num_iter, model_name="python"):
    """One BO cell. spec = ProblemSpec; model_cls = python Model class. Returns the SCHEMA v2 dict
    for np.savez: enough to re-analyze the run (raw replicates, initial block, true f/sigma at every
    sampled point, acquisition value + posterior at the recommended optimum) without re-running it."""
    t0 = time.time()
    bounds = spec.bounds                                 # (d, 2)
    d = spec.d

    ds = doe_cache.load(spec.name, seed, n_rep)          # SHARED initial design (CRN): identical for all 4 models
    Xs = ds["X_sample"]; Ys = ds["Y_sample"].copy(); Vs = ds["Var_sample"].copy()
    Y_rep_init = np.asarray(ds["Y_rep"], float)
    n_initial = Xs.shape[0]
    rng = np.random.default_rng([seed, 1])               # iteration-noise stream (distinct from the DOE's rng)

    data = {}
    for lv in spec.levels:
        m = Xs[:, d].astype(int) == lv                   # level = LAST column of [x_1..x_d, level]
        data[int(lv)] = dict(X=list(Xs[m, :d]), y_mean=list(Ys[m]), y_var=list(Vs[m]))

    X_sampled = list(map(list, Xs)); Y_sampled = list(Ys); Y_var_sampled = list(Vs)
    Y_rep_sampled = [list(row) for row in Y_rep_init]     # raw replicates, initial block first
    Y_min_history, Y_min_est, X_min_est = [], [], []
    X_next_history, Y_next_history, Y_var_next_history = [], [], []
    acf_val, mu_at_est, s_at_est, r_at_est = [], [], [], []
    needs_r = acquisitions.needs_aleatoric(acf)          # does the ACQUISITION consume r?

    for _ in range(num_iter):
        # Always fit the aleatoric poly (a cheap ridge): the acquisition only uses it when needs_r,
        # but r_at_est is recorded every iteration so the noise-at-best plot never needs a refit.
        model = model_cls.fit(data, needs_r=True)
        ymin = float(np.min(Y_sampled))

        mean_fn = lambda lv: (lambda xs: model.predict(lv, xs, observation_noise=False)[0])
        lv_est, x_est, y_est = _argbest_over_categories(mean_fn, model.levels, bounds)
        X_min_est.append(list(x_est) + [lv_est]); Y_min_est.append(y_est)

        xe = np.asarray(x_est, float).reshape(1, -1)      # posterior at the recommended optimum
        mu_e, s_e = model.mean_std(lv_est, xe)
        mu_at_est.append(float(np.ravel(mu_e)[0])); s_at_est.append(float(np.ravel(s_e)[0]))
        r_at_est.append(float(np.ravel(model.r(lv_est, xe))[0]))      # aleatoric VARIANCE

        def acq_fn(lv):
            def g(xs):
                mu, s = model.mean_std(lv, xs)
                r = model.r(lv, xs) if needs_r else None
                return acquisitions.evaluate(acf, mu, s, ymin, r=r, param=param)
            return g
        lv_next, x_next, u_next = _argbest_over_categories(acq_fn, model.levels, bounds)
        acf_val.append(float(u_next))                     # minimized U_negate at the chosen point

        y_rep = P.noisy_eval(spec, x_next, lv_next, n_rep, rng)
        y_mean, y_var = float(y_rep.mean()), float(y_rep.var(ddof=1))
        data[lv_next]["X"].append(np.asarray(x_next, float)); data[lv_next]["y_mean"].append(y_mean)
        data[lv_next]["y_var"].append(y_var)
        X_sampled.append(list(x_next) + [lv_next]); Y_sampled.append(y_mean); Y_var_sampled.append(y_var)
        Y_rep_sampled.append(list(np.asarray(y_rep, float)))
        X_next_history.append(list(x_next) + [lv_next]); Y_next_history.append(y_mean)
        Y_var_next_history.append(y_var)
        Y_min_history.append(float(np.min(Y_sampled)))

    Y_sampled_arr = np.asarray(Y_sampled)
    X_sampled_arr = np.asarray(X_sampled, float)
    bi = int(np.argmin(Y_sampled_arr))
    runtime = time.time() - t0

    # noise-free objective + true noise std at every sampled point -> the run stays re-scorable
    # even if problems.py is later edited.
    f_true = np.array([float(np.ravel(spec.f_true_level(r[:d], int(r[d])))[0]) for r in X_sampled_arr])
    sig_true = np.array([float(np.ravel(spec.sigma_level(r[:d], int(r[d])))[0]) for r in X_sampled_arr])

    return dict(
        Y_min_history=np.asarray(Y_min_history),
        X_sampled=X_sampled_arr,
        Y_sampled=Y_sampled_arr,
        Y_var_sampled=np.asarray(Y_var_sampled),
        Y_rep_sampled=np.asarray(Y_rep_sampled, float),          # (n_tr, n_rep) ALL raw replicates
        X_next_history=np.asarray(X_next_history, float),
        Y_next_history=np.asarray(Y_next_history),
        Y_var_next_history=np.asarray(Y_var_next_history),
        Y_min_est=np.asarray(Y_min_est),
        X_min_est=np.asarray(X_min_est, float),
        acf_val=np.asarray(acf_val),                             # acquisition value at the chosen point
        mu_at_est=np.asarray(mu_at_est),                         # posterior mean      @ X_min_est
        s_at_est=np.asarray(s_at_est),                           # epistemic std       @ X_min_est
        r_at_est=np.asarray(r_at_est),                           # aleatoric variance  @ X_min_est
        X_init=np.asarray(Xs, float), Y_init=np.asarray(Ys, float), Y_rep_init=Y_rep_init,
        f_true_sampled=f_true, sigma_true_sampled=sig_true,
        X_best_final=np.asarray(X_sampled[bi], float),
        Y_best_final=np.asarray([float(Y_sampled_arr[bi])]),
        Y_var_best_final=np.asarray([float(Y_var_sampled[bi])]),
        n_initial=np.asarray([n_initial]),
        meta=dict(problem=spec.name, model=model_name, acf=acf, acf_param=float(param),
                  n_rep=int(n_rep), seed=int(seed), num_iter=int(num_iter), runtime=float(runtime),
                  schema_version=2, doe_mode=P.DOE_MODE, n_init=int(spec.n_init),
                  n_levels=int(spec.n_levels), lb=float(spec.lb), ub=float(spec.ub),
                  timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                  # CRN holds for the INITIAL DESIGN only: python iteration noise is default_rng([seed,1]),
                  # MATLAB's is rng(seed). Seed-pairing across engines is exact on X_init/Y_init, not after.
                  noise_stream="python:default_rng([seed,1])"),
    )
