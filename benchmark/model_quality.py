"""
model_quality.py -- LOOP-FREE modeling comparison of the MATLAB heteroscedastic LVGP vs the native
Python port on branin_hetero. Both models are fit on the SAME data (each seed's final training set),
then scored on a dense (x, level) grid against the TRUE f and TRUE noise -- so this measures MODELING
quality, with none of the BO-loop / acquisition-optimizer / noise-stream confound that makes raw
regret misleading across engines.

Metrics (per model, averaged over seeds):
  * mean_rmse  -- RMSE of posterior mean vs true f              (surface accuracy; lower better)
  * noise_rmse -- RMSE of predicted aleatoric r(x) vs true var  (noise capture; lower better)
  * cov90      -- fraction of grid pts with |f_true-mu| <= 1.645*s (calibration; target 0.90)
  * sharp      -- mean posterior std (sharpness; lower better IF calibrated)
  * nll        -- mean Gaussian negative log predictive density of f_true (lower better; accuracy+calib)

MATLAB predictions are reconstructed from each cell's stored hyper (verified exact vs stored mu/s).
"""
import os, sys, glob, argparse
import numpy as np
from scipy.linalg import cholesky, solve_triangular
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import problems as P
import utils.models.hetero_lvgp_native as H
from utils.models.hetero_lvgp_native import HeterLVGPNative

NLV = 5


def _train(cell):
    X = np.asarray(cell["X_sampled"], float)
    return X[:, 0:1], X[:, -1].astype(int), np.asarray(cell["Y_sampled"], float).ravel(), \
        np.asarray(cell["Y_var_sampled"], float).ravel()


def matlab_model(cell):
    """Return predict(level,x)->(mu,s) and r(level,x) reconstructed from stored hyper."""
    h = cell["hyper"]; z = np.asarray(h["z"], float).reshape(NLV, 2); phi = float(np.ravel(h["phi"])[0])
    nug = float(h["nug_opt"]); sig2 = float(h["sigma2"])
    xq, lev, Y, v = _train(cell); lev0 = lev - 1
    xmin, xmax = xq.min(), xq.max(); span = (xmax - xmin) or 1.0; Xn = (xq - xmin) / span
    ymin, ymax = Y.min(), Y.max(); ysp = (ymax - ymin) or 1.0; Yn = ((Y - ymin) / ysp).reshape(-1, 1)
    scale2 = ysp * ysp
    Xtr = np.hstack([Xn, z[lev0]]); pf = np.concatenate([[phi], [0, 0]])
    Rc = H._corr(Xtr, Xtr, pf); Rc = 0.5 * (Rc + Rc.T); Rst = Rc + np.eye(len(Y)) * nug
    Rp = Rst + np.diag(v / scale2) / sig2
    L = cholesky(Rp, lower=True); k = len(Y); M = np.ones((k, 1))
    LiM = solve_triangular(L, M, lower=True); MTRinvM = float(LiM.T @ LiM)
    beta = float((LiM.T @ solve_triangular(L, Yn, lower=True)) / MTRinvM)
    temp = solve_triangular(L, Yn - M * beta, lower=True); RinvPY = solve_triangular(L.T, temp, lower=False)
    # aleatoric poly (recompute the feature standardization from training W, MATLAB-style)
    th = np.asarray(h["poly_theta"], float); Zl = np.asarray(h["poly_Z_latent"], float).reshape(NLV, 2)
    deg = int(h["poly_degree"])
    W = np.hstack([xq, Zl[lev0]]); muW = W.mean(0); sdW = W.std(0); sdW[sdW == 0] = 1.0

    def predict(level, x):
        xn = (np.atleast_2d(x) - xmin) / span
        Xnew = np.hstack([xn, np.repeat(z[level - 1][None, :], len(xn), 0)])
        Ron = H._corr(Xtr, Xnew, pf)
        yhat = (beta + Ron.T @ RinvPY).ravel() * ysp + ymin
        LiR = solve_triangular(L, Ron, lower=True); Wv = 1 - (LiM.T @ LiR)
        mse = sig2 * (1 - np.sum(LiR * LiR, 0) + (Wv.ravel() ** 2) / MTRinvM) * scale2
        return yhat, np.sqrt(np.maximum(mse, 1e-24))

    def r(level, x):
        Xx = np.atleast_2d(x)
        Wn = (np.hstack([Xx, np.repeat(Zl[level - 1][None, :], len(Xx), 0)]) - muW) / sdW
        return np.maximum(np.exp(2 * (H._poly_features(Wn, deg) @ th)), 1e-12)
    return predict, r


def _native_model(cell):
    xq, lev, Y, v = _train(cell); data = {}
    for L_ in range(1, NLV + 1):
        m = lev == L_
        if m.sum():
            data[L_] = {"X": xq[m], "y_mean": Y[m], "y_var": v[m]}
    mdl = HeterLVGPNative.fit(data, needs_r=True, bounds=P.get("branin_hetero").bounds)
    return (lambda L_, x: mdl.mean_std(L_, np.atleast_2d(x))), (lambda L_, x: mdl.r(L_, np.atleast_2d(x)))


def _metrics(predict, rfun, spec, grid):
    mu_e, f_e, s_e, r_e, sig_e = [], [], [], [], []
    for lv, xs in grid.items():
        mu, s = predict(lv, xs.reshape(-1, 1)); r = rfun(lv, xs.reshape(-1, 1))
        ft = np.array([float(np.ravel(spec.f_true_level(np.array([[x]]), lv))[0]) for x in xs])
        st = np.array([float(np.ravel(spec.sigma_level(np.array([[x]]), lv))[0]) for x in xs]) ** 2
        mu_e.append(mu); f_e.append(ft); s_e.append(np.maximum(s, 1e-9)); r_e.append(r); sig_e.append(st)
    mu = np.concatenate(mu_e); ft = np.concatenate(f_e); s = np.concatenate(s_e)
    r = np.concatenate(r_e); st = np.concatenate(sig_e)
    mean_rmse = np.sqrt(np.mean((mu - ft) ** 2))
    noise_rmse = np.sqrt(np.mean((r - st) ** 2))
    cov90 = np.mean(np.abs(ft - mu) <= 1.645 * s)
    nll = np.mean(0.5 * np.log(2 * np.pi * s ** 2) + 0.5 * ((ft - mu) / s) ** 2)
    return dict(mean_rmse=mean_rmse, noise_rmse=noise_rmse, cov90=cov90, sharp=s.mean(), nll=nll)


def compare(matlab_root="results", acq="lcb", nrep=10, nseed=15, ngrid=120):
    import scipy.io as sio
    spec = P.get("branin_hetero")
    xs = np.linspace(spec.lb, spec.ub, ngrid)
    grid = {lv: xs for lv in spec.levels}
    base = os.path.join(matlab_root, "branin_hetero", "heter_LVGP")
    sub = [d for d in os.listdir(base) if d == acq or d.startswith(acq + "_")][0]
    files = sorted(glob.glob(os.path.join(base, sub, f"nrep{nrep:02d}", "seed*.mat")))[:nseed]
    accM, accN = [], []
    for f in files:
        c = sio.loadmat(f, simplify_cells=True)
        try:
            pM, rM = matlab_model(c); pN, rN = _native_model(c)
            accM.append(_metrics(pM, rM, spec, grid)); accN.append(_metrics(pN, rN, spec, grid))
        except Exception as e:
            print("skip", os.path.basename(f), e)
    def agg(a): return {k: float(np.mean([d[k] for d in a])) for k in a[0]}
    return agg(accM), agg(accN), len(accM)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--acq", default="lcb"); ap.add_argument("--nseed", type=int, default=15)
    a = ap.parse_args()
    M, N, n = compare(acq=a.acq, nseed=a.nseed)
    print(f"\nMODELING QUALITY on branin_hetero (heter LVGP), {n} seeds, acq={a.acq}")
    print(f"{'metric':12s}{'MATLAB':>12s}{'Python native':>15s}{'winner':>10s}")
    lower = {"mean_rmse", "noise_rmse", "sharp", "nll"}
    for k in ["mean_rmse", "noise_rmse", "cov90", "sharp", "nll"]:
        win = ("Python" if (N[k] < M[k]) == (k in lower) else "MATLAB")
        if k == "cov90":  # closeness to 0.90
            win = "Python" if abs(N[k] - .9) < abs(M[k] - .9) else "MATLAB"
        print(f"{k:12s}{M[k]:12.4f}{N[k]:15.4f}{win:>10s}")
