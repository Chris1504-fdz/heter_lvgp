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


def _unwrap_cell(a):
    """Peel MATLAB 1x1 cell/object nesting down to the underlying array."""
    a = np.asarray(a)
    while a.dtype == object and a.size == 1:
        a = np.asarray(a.reshape(-1)[0])
    return a


def _load_cell(path):
    hyper_z = None                                            # LVGP latent embedding (matlab cells only)
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
        try:
            z = _unwrap_cell(d["hyper"][0, 0]["z"]).astype(float)
            hyper_z = z.reshape(-1, 2) if z.size else None    # (n_levels, dim_z=2)
        except Exception:
            hyper_z = None

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
        hyper_z=hyper_z,                                       # LVGP latent embedding (None for python)
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

# Per-problem deep dive grammar: ACQUISITION -> colour, MODEL -> (linestyle, marker). Kept consistent
# across every panel so a reader learns the encoding once. Acquisition colours follow CONFIG_ORDER.
ACQ_COLORS = {a: f"C{i}" for i, (a, _p) in enumerate(acquisitions.CONFIG_ORDER)}
MODEL_STYLES = {"standard_LVGP": ("-", "o"), "heter_LVGP": ("--", "s"),
                "separate_gp": (":", "^"), "categorical_kernel": ("-.", "D")}


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


def regret_heatmap(grid, functions=None, n_rep=10, ground_truth=True, configs=None,
                   normalize="row", annot_fmt="{:.3f}", figsize=None):
    """SINGLE consolidated heatmap of final regret. Rows = (problem x acquisition), columns = models;
    every cell is ANNOTATED with the mean final regret (value - f*) for that model/acq/problem.

    Because the problems differ by orders of magnitude in regret scale, a shared raw color scale would
    wash out. COLOR is therefore normalized per `normalize`:
      'row'  : within each (problem, acq) row, best model -> green (0), worst -> red (1). Answers
               "which model wins this row" at a glance; the printed number gives the magnitude.
      'func' : min-max within each PROBLEM (comparable colors across that problem's acquisitions).
      'none' : one raw scale (only sensible for a single problem).
    Cells with no data are left blank/grey. Returns the figure."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    fns = functions or grid.functions()
    cfgs = configs or acquisitions.CONFIG_ORDER
    models = _ordered_models(grid)
    base = _true_best_traj if ground_truth else _noisy_best_traj

    raw, ylabels, fkey, boundaries = [], [], [], []      # boundaries: y-index where a new problem starts
    for fn in fns:
        spec = P.get(fn); fstar = P.ground_truth_min(spec)
        started = False
        for (a, p) in cfgs:
            row = []
            for m in models:
                runs = grid.select(function=fn, model=m, acf=a, param=p, n_rep=n_rep)
                row.append(np.mean([base(r, spec)[-1] - fstar for r in runs]) if runs else np.nan)
            if np.all(np.isnan(row)):
                continue
            if not started:
                boundaries.append(len(raw)); started = True
            raw.append(row); fkey.append(fn)
            ylabels.append(f"{fn} · {acquisitions.label(a, p)}")
    if not raw:
        fig, ax = plt.subplots(figsize=(5, 2)); ax.text(0.5, 0.5, "no data yet", ha="center",
            va="center", color="crimson", transform=ax.transAxes); ax.axis("off"); return fig
    R = np.array(raw, float)                              # (rows, models) raw regret

    # colour matrix C in [0,1], best=0 -> green, worst=1 -> red, computed per the chosen scope
    C = np.full_like(R, np.nan)
    if normalize == "row":
        groups = [[i] for i in range(len(R))]
    elif normalize == "func":
        groups = [[i for i in range(len(R)) if fkey[i] == fn] for fn in dict.fromkeys(fkey)]
    else:
        groups = [list(range(len(R)))]
    for g in groups:
        block = R[g]; finite = block[np.isfinite(block)]
        lo, hi = (finite.min(), finite.max()) if finite.size else (0.0, 1.0)
        C[g] = 0.0 if hi <= lo else (block - lo) / (hi - lo)

    fig, ax = plt.subplots(figsize=figsize or (1.7 * len(models) + 3.0, 0.42 * len(R) + 1.6))
    im = ax.imshow(np.ma.masked_invalid(C), cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)
    ax.set_facecolor("0.85")                              # NaN cells show as grey
    ax.set_xticks(range(len(models))); ax.set_xticklabels([_mlabel(m) for m in models],
                                                          rotation=25, ha="right", fontsize=8)
    ax.set_yticks(range(len(R))); ax.set_yticklabels(ylabels, fontsize=7)
    for b in boundaries[1:]:                              # separators between problems
        ax.axhline(b - 0.5, color="k", lw=1.3)
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            if np.isfinite(R[i, j]):
                ax.text(j, i, annot_fmt.format(R[i, j]), ha="center", va="center", fontsize=7,
                        color="k" if 0.30 < C[i, j] < 0.85 else "w")
    ax.set_title(f"final regret (value − f*) per model / acq — n_rep={n_rep}, "
                 f"incumbent by {'true f' if ground_truth else 'observed mean'}\n"
                 f"colour = rank within each {'row' if normalize=='row' else normalize} "
                 f"(green = best model, red = worst); number = regret", fontsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_ticks([0, 1]); cb.set_ticklabels(["best", "worst"])
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


# ======================================================================================
#  PER-PROBLEM DEEP DIVE
#  A focused report for a single test problem: the landscape + initial DOE, per-model
#  convergence, the multi-acquisition overlay (acq=colour, model=style), the regret-vs-noise
#  trade-off, and the actual sampling trajectory. `problem_report` assembles + saves them;
#  `all_problem_reports` runs it for every problem into plots/<problem>/.
# ======================================================================================
def _one_run(grid, function, model, acf, param, n_rep, seed):
    for r in grid.runs:
        if (r["problem"] == function and r["model"] == model and r["acf"] == acf
                and r["n_rep"] == n_rep and r["seed"] == seed
                and (param != param or abs(r["param"] - param) < 1e-9)):
            return r
    return None


def _init_doe_run(grid, function, n_rep, seed):
    """Any cell for (function, n_rep, seed) carries the SHARED initial design (X_init/Y_rep_init)."""
    for r in grid.runs:
        if (r["problem"] == function and r["n_rep"] == n_rep and r["seed"] == seed
                and r.get("X_init") is not None):
            return r
    return None


def problem_landscape(function, grid=None, n_rep=10, seed=1, ax=None, show_doe=True):
    """The problem itself: noise-free f(x1|level) per category (coloured by level) with the true noise
    band f ± σ shaded, the global optimum starred, and -- if a matching cell exists -- the shared
    initial DOE (design points + their noisy replicates) overlaid so you see what every model started
    from."""
    import matplotlib.pyplot as plt
    spec = P.get(function)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    x = np.linspace(spec.lb, spec.ub, 400)
    for lv in spec.levels:
        c = plt.cm.tab10((lv - 1) % 10)
        f = spec.f_true_level(x, lv); sig = spec.sigma_level(x, lv)
        ax.plot(x, f, color=c, lw=1.6, label=f"level {lv}", zorder=3)
        ax.fill_between(x, f - sig, f + sig, color=c, alpha=0.10, zorder=1)   # true ±1σ noise band
    lv_opt, x_opt = P.true_opt_location(spec)
    ax.scatter([x_opt], [spec.f_true_level(x_opt, lv_opt)], marker="*", s=240, c="crimson",
               edgecolor="k", lw=0.6, zorder=6, label=f"opt (lv {lv_opt})")
    if show_doe and grid is not None:
        run = _init_doe_run(grid, function, n_rep, seed)
        if run is not None:
            Xi, Yr = run["X_init"], run["Y_rep_init"]
            for i in range(len(Xi)):
                c = plt.cm.tab10((int(Xi[i, 1]) - 1) % 10)
                ax.scatter(np.full(Yr.shape[1], Xi[i, 0]), Yr[i], s=10, color=c, alpha=0.35, zorder=4)
                ax.scatter([Xi[i, 0]], [Yr[i].mean()], s=55, color=c, edgecolor="k", lw=0.5, zorder=5)
    ax.set_xlabel("x1"); ax.set_ylabel("f")
    ax.set_title(f"{function}: landscape + initial DOE (seed {seed})", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=2)
    return ax


def noise_per_level(function, ax=None, as_variance=False, mark_opt=True):
    """The heteroscedastic noise structure: the TRUE noise std σ(x1│level) (or σ² with as_variance)
    for each categorical level over the domain, coloured to match the landscape. This is the quantity
    the noise-aware models try to learn -- where in (x1, level) space the objective is noisy."""
    import matplotlib.pyplot as plt
    spec = P.get(function)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    x = np.linspace(spec.lb, spec.ub, 400)
    for lv in spec.levels:
        c = plt.cm.tab10((lv - 1) % 10)
        s = spec.sigma_level(x, lv)
        ax.plot(x, s ** 2 if as_variance else s, color=c, lw=1.8, label=f"level {lv}", zorder=3)
    if mark_opt:
        lv_opt, x_opt = P.true_opt_location(spec)
        ax.axvline(x_opt, color="crimson", ls="--", lw=1, zorder=2,
                   label=f"optimum x1 (lv {lv_opt})")
    ax.set_xlabel("x1"); ax.set_ylabel("σ² true noise" if as_variance else "σ true noise std")
    ax.set_title(f"{function}: heteroscedastic noise per level", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=2)
    return ax


def convergence_overlay(grid, function, models=None, acqs=None, n_rep=10, ground_truth=True,
                        ax=None):
    """Every (model, acquisition) on ONE regret axis, using the shared grammar: ACQUISITION -> colour,
    MODEL -> line style + marker. Two legends (colour = acq, style = model), so you can read both
    factors at once -- e.g. 'does the noise-aware acq (one colour) help every model (every style)?'."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    spec = P.get(function); fstar = P.ground_truth_min(spec)
    base = _true_best_traj if ground_truth else _noisy_best_traj
    models = models or _ordered_models(grid)
    acqs = acqs or list(acquisitions.CONFIG_ORDER)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    seen_a, seen_m = set(), set()
    for (a, p) in acqs:
        col = ACQ_COLORS.get(a, "C7")
        for m in models:
            ls, mk = MODEL_STYLES.get(m, ("-", None))
            runs = grid.select(function=function, model=m, acf=a, param=p, n_rep=n_rep)
            if not runs:
                continue
            trajs = [base(r, spec) - fstar for r in runs]
            L = min(len(t) for t in trajs)
            mean = np.array([t[:L] for t in trajs]).mean(0)
            ax.plot(np.arange(1, L + 1), np.maximum(mean, 1e-6), color=col, ls=ls, marker=mk,
                    markevery=max(1, L // 8), ms=4, lw=1.5, alpha=0.9)
            seen_a.add((a, p)); seen_m.add(m)
    ax.set_yscale("log"); ax.set_xlabel("BO iteration"); ax.set_ylabel("regret = value − f*")
    ax.set_title(f"{function}: all models × acquisitions (n_rep={n_rep})", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ah = [Line2D([0], [0], color=ACQ_COLORS.get(a, "C7"), lw=2.5, label=acquisitions.label(a, p))
          for (a, p) in acqs if (a, p) in seen_a]
    mh = [Line2D([0], [0], color="0.3", ls=MODEL_STYLES.get(m, ("-", None))[0],
                 marker=MODEL_STYLES.get(m, ("-", None))[1], label=_mlabel(m))
          for m in models if m in seen_m]
    leg1 = ax.legend(handles=ah, title="acquisition (colour)", fontsize=7, title_fontsize=7,
                     loc="upper right")
    ax.add_artist(leg1)
    ax.legend(handles=mh, title="model (style)", fontsize=7, title_fontsize=7, loc="lower left")
    return ax


def tradeoff_scatter(grid, function, models=None, acqs=None, n_rep=10, ground_truth=True, ax=None):
    """Final regret (y) vs σ² at the incumbent (x), one point per (model, acquisition). Lower-LEFT is
    the sweet spot: accurate optimum in a quiet region. Colour = acquisition, marker = model -- the
    core heteroscedastic trade-off in a single view."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    spec = P.get(function); fstar = P.ground_truth_min(spec)
    base = _true_best_traj if ground_truth else _noisy_best_traj
    models = models or _ordered_models(grid)
    acqs = acqs or list(acquisitions.CONFIG_ORDER)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    seen_a, seen_m = set(), set()
    for (a, p) in acqs:
        col = ACQ_COLORS.get(a, "C7")
        for m in models:
            runs = grid.select(function=function, model=m, acf=a, param=p, n_rep=n_rep)
            if not runs:
                continue
            reg = np.mean([base(r, spec)[-1] - fstar for r in runs])
            s2 = np.mean([sigma2_at_best_traj(r, spec, ground_truth)[-1] for r in runs])
            ax.scatter([s2], [max(reg, 1e-6)], color=col, marker=MODEL_STYLES.get(m, ("-", "o"))[1],
                       s=70, edgecolor="k", lw=0.5, alpha=0.9, zorder=3)
            seen_a.add((a, p)); seen_m.add(m)
    ax.set_yscale("log")
    ax.set_xlabel("σ² at incumbent (lower = quieter)"); ax.set_ylabel("final regret (lower = better)")
    ax.set_title(f"{function}: regret vs noise trade-off (↙ best)", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ah = [Line2D([0], [0], color=ACQ_COLORS.get(a, "C7"), lw=0, marker="o", label=acquisitions.label(a, p))
          for (a, p) in acqs if (a, p) in seen_a]
    mh = [Line2D([0], [0], color="0.3", lw=0, marker=MODEL_STYLES.get(m, ("-", "o"))[1], label=_mlabel(m))
          for m in models if m in seen_m]
    leg1 = ax.legend(handles=ah, title="acq (colour)", fontsize=6.5, title_fontsize=7, loc="upper right")
    ax.add_artist(leg1)
    ax.legend(handles=mh, title="model (marker)", fontsize=6.5, title_fontsize=7, loc="lower right")
    return ax


def sampling_trajectory(grid, function, acf="ei", param=float("nan"), n_rep=10, seed=1,
                        models=None, axes=None):
    """Where each model actually SAMPLED: x1 location (y) per BO iteration (x), coloured by the chosen
    categorical level; the dashed red line is the optimum's x1, the grey line the DOE→BO boundary.
    Reveals exploration vs exploitation and whether the model locks onto the right category.
    One panel per model (needs `axes` with >= len(models) entries)."""
    import matplotlib.pyplot as plt
    spec = P.get(function); lv_opt, x_opt = P.true_opt_location(spec)
    models = models or _ordered_models(grid)
    for ax, m in zip(np.ravel(axes), models):
        run = _one_run(grid, function, m, acf, param, n_rep, seed)
        if run is None:
            ax.text(0.5, 0.5, "no data yet", ha="center", va="center", transform=ax.transAxes,
                    color="crimson"); ax.set_title(_mlabel(m), fontsize=9); continue
        X = run["X_sampled"]; n0 = run["n_initial"]
        it = np.arange(len(X)) - n0 + 1                       # <=0 initial DOE, >0 BO iterations
        cols = [plt.cm.tab10((int(l) - 1) % 10) for l in X[:, 1]]
        ax.scatter(it, X[:, 0], c=cols, s=20, edgecolor="k", lw=0.2, zorder=3)
        ax.axhline(x_opt, color="crimson", ls="--", lw=1, zorder=2)
        ax.axvline(0.5, color="grey", ls=":", lw=1, zorder=2)
        ax.set_xlabel("BO iteration"); ax.set_ylabel("x1")
        ax.set_ylim(spec.lb, spec.ub)
        ax.set_title(f"{_mlabel(m)}  (final regret "
                     f"{_true_best_traj(run, spec)[-1] - P.ground_truth_min(spec):.3f})", fontsize=9)
        ax.grid(alpha=0.3)
    return axes


def acquisition_facet(grid, function, models=None, n_rep=10, ground_truth=True, ncol=3,
                      share_y=True, noise_aware_only=False, ax_size=(4.7, 3.5)):
    """One panel PER ACQUISITION (from CONFIG_ORDER, those with data), each overlaying every model's
    regret convergence (mean ± s.e., log axis). Colour = model. With `share_y` all panels use ONE
    y-range (from the mean curves) so the acquisitions are directly comparable at a glance.
    `noise_aware_only=True` restricts to the noise-aware acquisitions (HAEI/ANPEI/RAHBO) instead of all 6."""
    import matplotlib.pyplot as plt
    spec = P.get(function); fstar = P.ground_truth_min(spec)
    base = _true_best_traj if ground_truth else _noisy_best_traj
    models = models or _ordered_models(grid)
    cfgs = [(a, p) for (a, p) in acquisitions.CONFIG_ORDER
            if (not noise_aware_only or acquisitions.needs_aleatoric(a))
            and any(grid.select(function=function, model=m, acf=a, param=p, n_rep=n_rep) for m in models)]
    if not cfgs:
        fig, ax = plt.subplots(figsize=(5, 2)); ax.text(0.5, 0.5, "no data yet", ha="center",
            va="center", color="crimson", transform=ax.transAxes); ax.axis("off"); return fig
    nrow = (len(cfgs) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(ax_size[0] * ncol, ax_size[1] * nrow), squeeze=False)
    for ax, (a, p) in zip(axes.ravel(), cfgs):
        _panel(ax, grid, function, a, p, n_rep, lambda r, s: base(r, s) - fstar,
               "regret (true) = value − f*", True)
        ax.set_title(acquisitions.label(a, p), fontsize=10)
    for k in range(len(cfgs), nrow * ncol):
        axes.ravel()[k].axis("off")
    if share_y:                                          # ONE y-range from the mean lines (ignore the
        lo = hi = None                                   # ±s.e. band floor so a wide band can't squash it)
        for ax in axes.ravel()[:len(cfgs)]:
            for ln in ax.get_lines():
                yd = np.asarray(ln.get_ydata(), float); yd = yd[np.isfinite(yd) & (yd > 0)]
                if yd.size:
                    lo = yd.min() if lo is None else min(lo, yd.min())
                    hi = yd.max() if hi is None else max(hi, yd.max())
        if lo and hi:
            for ax in axes.ravel()[:len(cfgs)]:
                ax.set_ylim(lo * 0.7, hi * 1.4)
    tag = "true f" if ground_truth else "observed mean"
    fig.suptitle(f"{function} — regret per acquisition (n_rep={n_rep}, incumbent by {tag}"
                 f"{'; shared y' if share_y else ''})", y=1.003, fontsize=12)
    fig.tight_layout()
    return fig


def problem_report(grid, function, acf="ei", param=float("nan"), n_rep=10, ground_truth=True,
                   seed=1, save_dir=None, show=True, dpi=110, noise_aware_only=False):
    """Full single-problem report -> two figures:
      overview.png     landscape+DOE · regret · value+f* · σ² · all-model×acq overlay · regret-vs-noise
      exploration.png  the sampling trajectory of each model (x1 vs iteration, coloured by level)
    `acf`/`param` fix the single acquisition used in the per-model convergence + exploration panels;
    the overlay and trade-off panels sweep ALL acquisitions. Saves under save_dir/ if given."""
    import os as _os
    import matplotlib.pyplot as plt
    spec = P.get(function)
    tag = "true f" if ground_truth else "observed mean"

    fig = plt.figure(figsize=(19, 9))
    axl = fig.add_subplot(2, 3, 1); problem_landscape(function, grid, n_rep, seed, ax=axl)
    ax2 = fig.add_subplot(2, 3, 2); noise_per_level(function, ax=ax2)   # heteroscedastic noise σ per level
    # (regret for this acquisition is still in panel 5's all-model×acq overlay and in by_acquisition.png)
    ax3 = fig.add_subplot(2, 3, 3)
    ok = _panel(ax3, grid, function, acf, param, n_rep,
                _true_best_traj if ground_truth else _noisy_best_traj, "best value", False)
    if ok:
        fstar = P.ground_truth_min(spec); ax3.axhline(fstar, color="grey", ls="--", lw=1.2)
        ax3.annotate(f"f* = {fstar:.3f}", xy=(0.98, fstar), xycoords=("axes fraction", "data"),
                     ha="right", va="bottom", fontsize=7, color="grey")
    ax3.set_title(f"{function}: value — {acquisitions.label(acf, param)}", fontsize=10)
    ax4 = fig.add_subplot(2, 3, 4)
    _panel(ax4, grid, function, acf, param, n_rep,
           lambda r, s: sigma2_at_best_traj(r, s, ground_truth), "σ² at incumbent", True)
    ax4.set_title(f"{function}: σ² at incumbent — {acquisitions.label(acf, param)}", fontsize=10)
    convergence_overlay(grid, function, n_rep=n_rep, ground_truth=ground_truth,
                        ax=fig.add_subplot(2, 3, 5))
    tradeoff_scatter(grid, function, n_rep=n_rep, ground_truth=ground_truth,
                     ax=fig.add_subplot(2, 3, 6))
    fig.suptitle(f"{function} — per-problem report (n_rep={n_rep}, incumbent by {tag})",
                 fontsize=13, y=1.005)
    fig.tight_layout()

    models = _ordered_models(grid)
    ncol = 2; nrow = (len(models) + ncol - 1) // ncol
    fig2, axes2 = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 4.2 * nrow), squeeze=False)
    sampling_trajectory(grid, function, acf, param, n_rep, seed, models, axes2)
    for k in range(len(models), nrow * ncol):
        axes2.ravel()[k].axis("off")
    from matplotlib.lines import Line2D                          # legend mapping colour -> level
    lv_opt, _ = P.true_opt_location(spec)
    lv_handles = [Line2D([0], [0], color=plt.cm.tab10((lv - 1) % 10), lw=0, marker="o", ms=6,
                         label=f"level {lv}" + ("  (opt)" if lv == lv_opt else "")) for lv in spec.levels]
    lv_handles.append(Line2D([0], [0], color="crimson", ls="--", lw=1.2, label="optimum x1"))
    fig2.legend(handles=lv_handles, loc="lower center", ncol=len(lv_handles), fontsize=8,
                frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig2.suptitle(f"{function} — sampling trajectory · {acquisitions.label(acf, param)}, seed {seed} "
                  f"(colour = chosen level)", fontsize=12, y=1.01)
    fig2.tight_layout(rect=(0, 0.03, 1, 1))

    fig3 = acquisition_facet(grid, function, models=models, n_rep=n_rep, ground_truth=ground_truth,
                             noise_aware_only=noise_aware_only)
    fig4 = latent_space(grid, function, acf=acf, param=param, n_rep=n_rep, seed=seed)

    if save_dir:
        _os.makedirs(save_dir, exist_ok=True)
        fig.savefig(_os.path.join(save_dir, "overview.png"), dpi=dpi, bbox_inches="tight")
        fig2.savefig(_os.path.join(save_dir, "exploration.png"), dpi=dpi, bbox_inches="tight")
        fig3.savefig(_os.path.join(save_dir, "by_acquisition.png"), dpi=dpi, bbox_inches="tight")
        fig4.savefig(_os.path.join(save_dir, "latent_space.png"), dpi=dpi, bbox_inches="tight")
    if not show:
        plt.close(fig); plt.close(fig2); plt.close(fig3); plt.close(fig4)
    return fig, fig2, fig3, fig4


def all_problem_reports(grid, functions=None, save_root="plots", acf="ei", param=float("nan"),
                        n_rep=10, ground_truth=True, seed=1, show=False, noise_aware_only=False):
    """Run problem_report for every problem into
    save_root/<problem>/{overview,exploration,by_acquisition}.png. Returns the dirs written.
    `show=False` keeps notebook output clean for a batch."""
    import os as _os
    fns = functions or grid.functions()
    out = []
    for fn in fns:
        d = _os.path.join(save_root, fn)
        problem_report(grid, fn, acf=acf, param=param, n_rep=n_rep, ground_truth=ground_truth,
                       seed=seed, save_dir=d, show=show, noise_aware_only=noise_aware_only)
        out.append(d)
        print(f"  wrote {d}/overview.png + exploration.png + by_acquisition.png + latent_space.png")
    return out


# ======================================================================================
#  LVGP LATENT SPACE vs GROUND TRUTH
#  LVGP maps each categorical level to a 2-D latent point z; levels that behave similarly
#  under the true objective should land close together. We compare the learned geometry to
#  the TRUE geometry (how different the levels' noise-free curves actually are).
#  Only the LVGP models expose z (matlab `hyper.z`); python models have no such embedding.
# ======================================================================================
LVGP_MODELS = ("standard_LVGP", "heter_LVGP")


def _level_curve_dist(spec, n=300):
    """Ground-truth distance between categorical levels = RMS gap between their noise-free curves
    f(x1│level) over the domain. This is the geometry a good latent embedding should reproduce."""
    x = np.linspace(spec.lb, spec.ub, n)
    F = np.array([spec.f_true_level(x, lv) for lv in spec.levels])       # (L, n)
    L = len(spec.levels)
    D = np.zeros((L, L))
    for i in range(L):
        for j in range(L):
            D[i, j] = np.sqrt(np.mean((F[i] - F[j]) ** 2))
    return D


def _latent_runs(grid, function, model, acf, param, n_rep):
    return [r for r in grid.runs if r["problem"] == function and r["model"] == model
            and r["acf"] == acf and r["n_rep"] == n_rep and r.get("hyper_z") is not None
            and (param != param or abs(r["param"] - param) < 1e-9)]


def latent_space(grid, function, models=None, acf="ei", param=float("nan"), n_rep=10, seed=1):
    """Compare the LVGP LATENT EMBEDDING of the categorical levels with the ground truth.
    Per LVGP model: (1) a latent map for one seed — each level's learned 2-D z, annotated, coloured by
    its true underlying value (spec.meta['cat_values']); (2) a recovery panel pooling ALL seeds —
    learned latent distance vs the true curve distance between level pairs, with a Spearman ρ (rank
    correlation, invariant to the rotation/reflection/scale ambiguity of the embedding). ρ→1 means the
    latent geometry reproduces the true category geometry."""
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr
    spec = P.get(function)
    L = spec.n_levels
    catv = np.asarray(spec.meta.get("cat_values", np.arange(1, L + 1)), float)
    lv_opt, _ = P.true_opt_location(spec)
    Dtrue = _level_curve_dist(spec)
    iu = np.triu_indices(L, k=1)
    dtrue = Dtrue[iu]
    models = [m for m in (models or LVGP_MODELS) if _latent_runs(grid, function, m, acf, param, n_rep)]
    if not models:
        fig, ax = plt.subplots(figsize=(5, 2)); ax.text(0.5, 0.5, "no LVGP latent data yet",
            ha="center", va="center", color="crimson", transform=ax.transAxes); ax.axis("off"); return fig

    ncol = len(models) + 1
    fig, axes = plt.subplots(1, ncol, figsize=(4.9 * ncol, 4.6), squeeze=False)
    axes = axes.ravel()
    for k, m in enumerate(models):
        runs = _latent_runs(grid, function, m, acf, param, n_rep)
        seed_run = next((r for r in runs if r["seed"] == seed), runs[0])
        z = np.asarray(seed_run["hyper_z"], float)
        ax = axes[k]
        sc = ax.scatter(z[:, 0], z[:, 1], c=catv, cmap="viridis", s=180, edgecolor="k", lw=0.6, zorder=3)
        for lv in range(L):
            ax.annotate(f"lv{lv+1}" + ("★" if lv + 1 == lv_opt else ""), (z[lv, 0], z[lv, 1]),
                        fontsize=8, ha="center", va="center",
                        color="w" if catv[lv] < catv.mean() else "k", zorder=4)
        cb = fig.colorbar(sc, ax=ax, fraction=0.046); cb.set_label("true level value", fontsize=8)
        ax.set_xlabel("latent z₁"); ax.set_ylabel("latent z₂")
        ax.set_title(f"{_mlabel(m)} — latent map (seed {seed_run['seed']})", fontsize=10)
        ax.grid(alpha=0.3); ax.set_aspect("equal", adjustable="datalim")

    # RANK-vs-RANK recovery. The LVGP embedding is identifiable only up to rotation/reflection/SCALE,
    # so an isometry (y = x on raw distances) is not a meaningful target -- only the ORDER of the
    # pairwise distances is. Plotting rank(true) vs rank(latent) makes the y = x diagonal mean exactly
    # "perfect Spearman ρ = 1": points on it = pairs ordered correctly, points off it = pairs the
    # embedding put out of order.
    from scipy.stats import rankdata
    axr = axes[-1]
    rt = rankdata(dtrue)
    npair = len(dtrue)
    axr.plot([0.5, npair + 0.5], [0.5, npair + 0.5], "k--", lw=1.3, zorder=1,
             label="perfect fidelity (ρ = 1)")
    for m in models:
        runs = _latent_runs(grid, function, m, acf, param, n_rep)
        dz_pool, rhos = [], []
        for r in runs:
            z = np.asarray(r["hyper_z"], float)
            Dz = np.sqrt(((z[:, None, :] - z[None, :, :]) ** 2).sum(-1))[iu]
            if Dz.max() > 0:
                dz_pool.append(Dz)
                rhos.append(spearmanr(Dz, dtrue).statistic)     # rank corr, invariant to the scale ambiguity
        if not dz_pool:
            continue
        rz = rankdata(np.mean(dz_pool, 0))
        col = MODEL_COLORS.get(m, "C7")
        axr.scatter(rt, rz, color=col, s=55, edgecolor="k", lw=0.4, zorder=3,
                    label=f"{_mlabel(m)}  ρ={np.mean(rhos):.2f}±{np.std(rhos):.2f}")
    axr.set_xlabel("rank of TRUE distance between levels")
    axr.set_ylabel("rank of LEARNED latent distance")
    axr.set_title(f"latent vs true geometry (rank–rank)\ndiagonal = faithful ordering; "
                  f"Spearman ρ over {len(runs)} seeds", fontsize=10)
    axr.set_aspect("equal", adjustable="box"); axr.grid(alpha=0.3); axr.legend(fontsize=8)
    fig.suptitle(f"{function} — LVGP latent space vs ground truth "
                 f"({acquisitions.label(acf, param)}, n_rep={n_rep})", y=1.02, fontsize=12)
    fig.tight_layout()
    return fig
