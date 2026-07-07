"""
results.py -- load and compare the (function x model) grid. Reads BOTH .npz (python engine) and
.mat (matlab engine) cells from results/<function>/<model>/<acq>/nrep<NN>/seed<NN>, indexed by
(function, model, acf, param, n_rep, seed). The headline metric is `true_best_sampled` (min f_true
over the sampled points), consistent with the study_v2* studies.
"""
import os
import glob
import numpy as np

from . import problems as P
from . import acquisitions
from .models import MODELS


def _load_cell(path):
    if path.endswith(".npz"):
        m = np.load(path, allow_pickle=True)
        meta = m["meta"].item()
        g = lambda k: np.asarray(m[k])
    else:
        import scipy.io
        d = scipy.io.loadmat(path)
        mm = d["meta"][0, 0]
        meta = {}
        for k in mm.dtype.names:
            v = mm[k]
            meta[k] = str(v[0]) if v.dtype.kind in "US" else float(np.ravel(v)[0])
        g = lambda k: np.asarray(d[k])
    return dict(
        problem=str(meta.get("problem")), model=str(meta.get("model")), acf=str(meta.get("acf")),
        param=float(meta.get("acf_param", float("nan"))), n_rep=int(float(meta.get("n_rep"))),
        seed=int(float(meta.get("seed"))), runtime=float(meta.get("runtime", 0.0)),
        Y_min_history=np.ravel(g("Y_min_history")).astype(float),
        X_sampled=g("X_sampled").astype(float).reshape(-1, 2),
        X_min_est=g("X_min_est").astype(float).reshape(-1, 2),
        Y_var_sampled=np.ravel(g("Y_var_sampled")).astype(float),
        n_initial=int(np.ravel(g("n_initial"))[0]),
    )


class GridResults:
    def __init__(self, root, runs):
        self.root = root
        self.runs = runs

    @classmethod
    def load(cls, root="results"):
        runs = []
        for f in (glob.glob(os.path.join(root, "**", "*.npz"), recursive=True)
                  + glob.glob(os.path.join(root, "**", "*.mat"), recursive=True)):
            try:
                runs.append(_load_cell(f))
            except Exception as e:
                print(f"  skip {f}: {e}")
        return cls(root, runs)

    def functions(self):
        return sorted({r["problem"] for r in self.runs})

    def models(self):
        return sorted({r["model"] for r in self.runs})

    def select(self, function=None, model=None, acf=None, param=None, n_rep=None):
        out = self.runs
        if function is not None: out = [r for r in out if r["problem"] == function]
        if model is not None:    out = [r for r in out if r["model"] == model]
        if acf is not None:      out = [r for r in out if r["acf"] == acf]
        if n_rep is not None:    out = [r for r in out if r["n_rep"] == n_rep]
        if param is not None and param == param:
            out = [r for r in out if abs(r["param"] - param) < 1e-9]
        return out


def _true_best_traj(run, spec):
    """Per-iteration true_best_sampled = cumulative min f_true over sampled points."""
    X = run["X_sampled"]
    ft = np.array([float(spec.f_true_level(x[0], int(round(x[1])))) for x in X])
    cummin = np.minimum.accumulate(ft)
    n0 = run["n_initial"]
    niter = len(run["Y_min_history"])
    idx = np.clip(n0 - 1 + np.arange(1, niter + 1), 0, len(ft) - 1)
    return cummin[idx]


def compare_models_on_function(grid, function, acf="ei", param=float("nan"), n_rep=10,
                               as_regret=True, logy=True, ax=None):
    """Overlay each model's true-value convergence (mean +/- s.e. over seeds) for one (function,
    acquisition, n_rep). as_regret -> value - f*; logy -> log axis. Returns the axes."""
    import matplotlib.pyplot as plt
    spec = P.get(function)
    fstar = P.ground_truth_min(spec)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    colors = ["C0", "C3", "C2", "C1", "C4", "C5"]
    for m, col in zip(sorted(grid.models()), colors):
        runs = grid.select(function=function, model=m, acf=acf, param=param, n_rep=n_rep)
        if not runs:
            continue
        trajs = [_true_best_traj(r, spec) for r in runs]
        L = min(len(t) for t in trajs)
        A = np.array([t[:L] for t in trajs])
        A = (A - fstar) if as_regret else A
        mean = A.mean(0); sem = A.std(0, ddof=1) / np.sqrt(len(A)) if len(A) > 1 else np.zeros(L)
        lab = MODELS[m].label if m in MODELS else m
        x = np.arange(1, L + 1)
        m_ = np.maximum(mean, 1e-6) if (logy and as_regret) else mean
        ax.plot(x, m_, color=col, lw=2, label=f"{lab} (n={len(A)})")
        ax.fill_between(x, np.maximum(mean - sem, 1e-6) if (logy and as_regret) else mean - sem,
                        mean + sem, color=col, alpha=0.15)
    if not as_regret:
        ax.axhline(fstar, color="grey", ls="--", lw=1, label=f"f* = {fstar:.4f}")
    if logy and as_regret:
        ax.set_yscale("log")
    ax.set_xlabel("BO iteration")
    ax.set_ylabel("regret = value − f*" if as_regret else "best true value")
    ax.set_title(f"{function} — {acquisitions.label(acf, param)} (n_rep={n_rep})")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    return ax


def final_regret_table(grid, acf="ei", param=float("nan"), n_rep=10):
    """Mean +/- s.e. final true_best regret per (function, model) for one acquisition."""
    import pandas as pd
    rows = {}
    for fn in grid.functions():
        spec = P.get(fn); fstar = P.ground_truth_min(spec)
        row = {}
        for m in grid.models():
            runs = grid.select(function=fn, model=m, acf=acf, param=param, n_rep=n_rep)
            if not runs:
                row[MODELS[m].label if m in MODELS else m] = None; continue
            fr = np.array([_true_best_traj(r, spec)[-1] - fstar for r in runs])
            row[MODELS[m].label if m in MODELS else m] = f"{fr.mean():.3f} ± {fr.std():.3f}"
        rows[fn] = row
    return pd.DataFrame(rows).T
