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

    def opt(k, flat=True):
        """A schema-v2 field, or None on a v1 cell that predates it."""
        try:
            v = np.asarray(g(k), float)
        except Exception:
            return None
        return np.ravel(v) if flat else v

    out = dict(
        problem=str(meta.get("problem")), model=str(meta.get("model")), acf=str(meta.get("acf")),
        param=float(meta.get("acf_param", float("nan"))), n_rep=int(float(meta.get("n_rep"))),
        seed=int(float(meta.get("seed"))), runtime=float(meta.get("runtime", 0.0)),
        schema_version=int(float(meta.get("schema_version", 1))),
        Y_min_history=np.ravel(g("Y_min_history")).astype(float),
        X_sampled=g("X_sampled").astype(float).reshape(-1, 2),
        Y_sampled=np.ravel(g("Y_sampled")).astype(float),      # replicate MEAN at each sampled point
        X_min_est=g("X_min_est").astype(float).reshape(-1, 2),
        Y_var_sampled=np.ravel(g("Y_var_sampled")).astype(float),
        n_initial=int(np.ravel(g("n_initial"))[0]),
    )
    out.update(                                               # ---- schema v2 (None on a v1 cell) ----
        Y_rep_sampled=opt("Y_rep_sampled", flat=False),       # (n_tr, n_rep) raw replicates
        acf_val=opt("acf_val"),                               # acquisition value at the chosen point
        mu_at_est=opt("mu_at_est"), s_at_est=opt("s_at_est"), r_at_est=opt("r_at_est"),
        f_true_sampled=opt("f_true_sampled"), sigma_true_sampled=opt("sigma_true_sampled"),
        X_init=opt("X_init", flat=False), Y_init=opt("Y_init"),
        Y_rep_init=opt("Y_rep_init", flat=False),
    )
    return out


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


# ======================================================================================
#  Multi-function views (ports the previous_ver/compare_methods.ipynb plots to the grid).
#  Every function below takes the FULL grid and facets over `functions` -- so the same call
#  works for 1 or all 10 test problems.
#
#  GROUND_TRUTH toggle, as in the old notebook:
#    True  -> the incumbent is chosen by the NOISE-FREE f, and sigma^2 is the TRUE noise variance
#    False -> the incumbent is chosen by the observed replicate mean, and sigma^2 is its sample variance
#  The `_true_best_traj` above is the ground-truth trajectory; `_noisy_best_traj` is its counterpart.
# ======================================================================================
MODEL_COLORS = {"standard_LVGP": "C0", "heter_LVGP": "C3",
                "separate_gp": "C2", "categorical_kernel": "C1"}
MODEL_ORDER = ["standard_LVGP", "heter_LVGP", "separate_gp", "categorical_kernel"]


def _mlabel(m):
    return MODELS[m].label if m in MODELS else m


def _ordered_models(grid):
    present = set(grid.models())
    return [m for m in MODEL_ORDER if m in present] + sorted(present - set(MODEL_ORDER))


def _iter_slice(run, arr):
    """Restrict a per-SAMPLED-POINT array to the BO iterations (drop the initial DOE block)."""
    n0, niter = run["n_initial"], len(run["Y_min_history"])
    idx = np.clip(n0 - 1 + np.arange(1, niter + 1), 0, len(arr) - 1)
    return arr[idx], idx


def _noisy_best_traj(run, spec):
    """Cumulative min of the OBSERVED replicate mean (what the optimizer actually sees)."""
    return _iter_slice(run, np.minimum.accumulate(run["Y_sampled"]))[0]


def _incumbent_idx(run, spec, ground_truth):
    """Index (into X_sampled) of the incumbent best design after each BO iteration."""
    X = run["X_sampled"]
    if ground_truth:
        vals = run["f_true_sampled"]
        if vals is None:                                   # v1 cell -> recompute from the spec
            vals = np.array([float(spec.f_true_level(x[0], int(round(x[1])))) for x in X])
    else:
        vals = run["Y_sampled"]
    argmins = np.empty(len(vals), int)                     # running argmin (no float-equality tricks)
    cur = 0
    for i, v in enumerate(vals):
        if v < vals[cur]:
            cur = i
        argmins[i] = cur
    return _iter_slice(run, argmins)[0]


def sigma2_at_best_traj(run, spec, ground_truth=True):
    """Noise VARIANCE at the incumbent best design, per BO iteration.
    ground_truth -> the true sigma(x,level)^2; else the replicate sample variance actually observed."""
    idx = _incumbent_idx(run, spec, ground_truth)
    if ground_truth:
        sig = run["sigma_true_sampled"]
        if sig is None:
            X = run["X_sampled"]
            sig = np.array([float(spec.sigma_level(x[0], int(round(x[1])))) for x in X])
        return sig[idx] ** 2
    return run["Y_var_sampled"][idx]


def _panel(ax, grid, function, acf, param, n_rep, series_fn, ylabel, logy, title_extra=""):
    """Shared plumbing: overlay every model's mean +/- s.e. trajectory on one axes."""
    spec = P.get(function)
    any_data = False
    for m in _ordered_models(grid):
        runs = grid.select(function=function, model=m, acf=acf, param=param, n_rep=n_rep)
        if not runs:
            continue
        trajs = [series_fn(r, spec) for r in runs]
        L = min(len(t) for t in trajs)
        A = np.array([t[:L] for t in trajs])
        mean = A.mean(0)
        sem = A.std(0, ddof=1) / np.sqrt(len(A)) if len(A) > 1 else np.zeros(L)
        x = np.arange(1, L + 1)
        lo = np.maximum(mean - sem, 1e-12) if logy else mean - sem
        ax.plot(x, np.maximum(mean, 1e-12) if logy else mean,
                color=MODEL_COLORS.get(m, "C7"), lw=2, label=f"{_mlabel(m)} (n={len(A)})")
        ax.fill_between(x, lo, mean + sem, color=MODEL_COLORS.get(m, "C7"), alpha=0.15)
        any_data = True
    if logy:
        ax.set_yscale("log")
    if not any_data:
        ax.text(0.5, 0.5, "no data yet", ha="center", va="center", transform=ax.transAxes,
                color="crimson", fontsize=11)
    ax.set_xlabel("BO iteration"); ax.set_ylabel(ylabel)
    ax.set_title(f"{function}{title_extra}", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    if any_data:
        ax.legend(fontsize=7)
    return any_data


def facet(grid, kind, functions=None, acf="ei", param=float("nan"), n_rep=10,
          ground_truth=True, logy=True, ncol=2, figsize=(6.4, 4.4)):
    """One panel per test function, models overlaid. kind:
         'regret'   value - f*            (log axis when logy)
         'value'    best true value, with the f* horizontal line
         'sigma2'   noise variance at the incumbent best design
    """
    import matplotlib.pyplot as plt
    fns = functions or grid.functions()
    nrow = (len(fns) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(figsize[0] * ncol, figsize[1] * nrow), squeeze=False)
    tag = "true f" if ground_truth else "observed mean"
    for ax, fn in zip(axes.ravel(), fns):
        spec = P.get(fn)
        fstar = P.ground_truth_min(spec)
        if kind == "regret":
            base = _true_best_traj if ground_truth else _noisy_best_traj
            _panel(ax, grid, fn, acf, param, n_rep, lambda r, s: base(r, s) - fstar,
                   "regret = value − f*", logy)
        elif kind == "value":
            base = _true_best_traj if ground_truth else _noisy_best_traj
            ok = _panel(ax, grid, fn, acf, param, n_rep, base, "best value", False)
            if ok:
                ax.axhline(fstar, color="grey", ls="--", lw=1.2)
                ax.annotate(f"f* = {fstar:.3f}", xy=(0.98, fstar), xycoords=("axes fraction", "data"),
                            ha="right", va="bottom", fontsize=7, color="grey")
        elif kind == "sigma2":
            _panel(ax, grid, fn, acf, param, n_rep,
                   lambda r, s: sigma2_at_best_traj(r, s, ground_truth),
                   "σ² at incumbent", logy)
        else:
            raise ValueError(f"unknown kind {kind!r}")
    for k in range(len(fns), nrow * ncol):
        axes.ravel()[k].axis("off")
    fig.suptitle(f"{kind} — {acquisitions.label(acf, param)}, n_rep={n_rep}, incumbent by {tag}",
                 y=1.002, fontsize=12)
    fig.tight_layout()
    return fig


def heatmaps_by_function(grid, functions=None, n_rep=10, ground_truth=True, ncol=2,
                         configs=None, annot_fmt="{:.2f}"):
    """One heatmap per function: final regret for every (acquisition x model) actually present.
    Rows with no data anywhere are dropped, so there are no blank bands."""
    import matplotlib.pyplot as plt
    fns = functions or grid.functions()
    cfgs = configs or acquisitions.CONFIG_ORDER
    models = _ordered_models(grid)
    nrow = (len(fns) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(1.9 * len(models) + 3.2, 2.6 * nrow), squeeze=False)
    for ax, fn in zip(axes.ravel(), fns):
        spec = P.get(fn); fstar = P.ground_truth_min(spec)
        base = _true_best_traj if ground_truth else _noisy_best_traj
        rows, labels = [], []
        for (a, p) in cfgs:
            row = []
            for m in models:
                runs = grid.select(function=fn, model=m, acf=a, param=p, n_rep=n_rep)
                row.append(np.mean([base(r, spec)[-1] - fstar for r in runs]) if runs else np.nan)
            if not np.all(np.isnan(row)):
                rows.append(row); labels.append(acquisitions.label(a, p))
        if not rows:
            ax.text(0.5, 0.5, "no data yet", ha="center", va="center", transform=ax.transAxes,
                    color="crimson"); ax.set_title(fn, fontsize=10); ax.axis("off"); continue
        M = np.array(rows)
        im = ax.imshow(M, cmap="viridis_r", aspect="auto")
        ax.set_xticks(range(len(models))); ax.set_xticklabels([_mlabel(m) for m in models],
                                                              rotation=30, ha="right", fontsize=7)
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=7)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, annot_fmt.format(M[i, j]), ha="center", va="center", fontsize=6,
                            color="w" if M[i, j] > np.nanmean(M) else "k")
        ax.set_title(f"{fn}  (final regret, n_rep={n_rep})", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)
    for k in range(len(fns), nrow * ncol):
        axes.ravel()[k].axis("off")
    fig.tight_layout()
    return fig


def coverage(grid, n_rep=None):
    """How many seeds exist per (function, model, acquisition) -- the sweep is long, so know what
    the plots above are actually averaging over."""
    import pandas as pd
    rows = []
    for r in grid.runs:
        if n_rep is not None and r["n_rep"] != n_rep:
            continue
        rows.append(dict(function=r["problem"], model=_mlabel(r["model"]),
                         acq=acquisitions.label(r["acf"], r["param"]), seed=r["seed"]))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.pivot_table(index=["function", "acq"], columns="model", values="seed",
                          aggfunc="count", fill_value=0)


def runtime_table(grid):
    """Mean wall-clock seconds per cell, function x model."""
    import pandas as pd
    rows = []
    for r in grid.runs:
        rows.append(dict(function=r["problem"], model=_mlabel(r["model"]), runtime=r["runtime"]))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).pivot_table(index="function", columns="model",
                                          values="runtime", aggfunc="mean").round(1)
