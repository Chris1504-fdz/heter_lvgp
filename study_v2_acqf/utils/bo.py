"""
bo.py -- BO loop for study_v2_acqf: joint categorical GP + BoTorch `optimize_acqf_mixed`.

Same problem / DOE / saved schema as the other studies, but the acquisition is optimized with
BoTorch's `optimize_acqf_mixed` (enumerate the 5 categorical codes via fixed_features_list,
gradient-optimize x1 within each) instead of the hand-rolled grid + L-BFGS. The model is fit on
-y (maximize convention, see model.py); EI/LCB/PI use BoTorch built-ins and haei/anpei/rahbo are
custom AnalyticAcquisitionFunction subclasses (see acqf.py).
"""
import time
import numpy as np
import torch

from botorch.optim import optimize_acqf_mixed
from botorch.acquisition.analytic import PosteriorMean

from . import problem
from .aleatoric import AleatoricModels
from .model import fit_neg_mixed_gp
from .acqf import make_acqf, TorchAleatoric

DTYPE = torch.float64
NUM_RESTARTS = 10
RAW_SAMPLES = 256        # grid-like init: het-aware acqs (haei/anpei) are near-flat (r >> s^2),
                         # so a dense raw-sample sweep is what finds their argmax (L-BFGS can't climb flat)
_NEEDS_ALE = {"haei", "anpei", "rahbo"}


def _optimize_mixed(acq, bounds, fixed_features):
    cand, val = optimize_acqf_mixed(acq, bounds=bounds, q=1, num_restarts=NUM_RESTARTS,
                                    raw_samples=RAW_SAMPLES, fixed_features_list=fixed_features)
    x1 = float(cand[0, 0]); level = int(round(float(cand[0, 1]))) + 1     # code -> 1-based level
    return x1, level, float(val)


def run_bo(acf, param, n_rep, seed, num_iter):
    """Run one BO cell. Returns a dict of numpy arrays + meta, ready to np.savez."""
    t0 = time.time()
    rng = np.random.default_rng(seed)
    doe = problem.initial_doe(n_rep, rng=rng)
    Xs = doe["X_sample"]; Ys = doe["Y_sample"].copy(); Vs = doe["Var_sample"].copy()
    n_initial = Xs.shape[0]

    data = {}
    for lv in problem.LEVELS:
        m = Xs[:, 1].astype(int) == lv
        data[int(lv)] = dict(x1=list(Xs[m, 0]), y_mean=list(Ys[m]), y_var=list(Vs[m]))

    X_sampled = list(map(list, Xs)); Y_sampled = list(Ys); Y_var_sampled = list(Vs)
    Y_min_history, Y_min_est, X_min_est = [], [], []
    X_next_history, Y_next_history, Y_var_next_history = [], [], []

    bounds = torch.tensor([[problem.LB, 0.0], [problem.UB, float(problem.N_LV - 1)]], dtype=DTYPE)
    fixed_features = [{1: float(c)} for c in range(problem.N_LV)]
    needs_ale = acf in _NEEDS_ALE

    for _ in range(num_iter):
        model = fit_neg_mixed_gp(data)                       # fit on -y (maximize)
        best_f = float(-np.min(Y_sampled))                   # max(-y_obs) = -min(y_obs)
        tale = TorchAleatoric(AleatoricModels.fit(data), problem.LEVELS) if needs_ale else None

        # recommended optimum: argmax posterior mean of -f = argmin mu_f
        x_est, lv_est, val_est = _optimize_mixed(PosteriorMean(model), bounds, fixed_features)
        X_min_est.append([x_est, lv_est]); Y_min_est.append(-val_est)    # min mu_f = -max(-mu_f)

        # next point: argmax of the acquisition
        acq = make_acqf(acf, param, model, best_f, tale)
        x_next, lv_next, _ = _optimize_mixed(acq, bounds, fixed_features)

        y_rep = problem.noisy_eval(x_next, lv_next, n_rep, rng)
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
        var_fctr=problem.VAR_FCTR,
        meta=dict(acf=acf, acf_param=float(param), n_rep=int(n_rep), seed=int(seed),
                  num_iter=int(num_iter), runtime=float(runtime), model="botorch_acqf"),
    )
