"""
plot_utils.py — plotting utilities for the heteroscedastic LVGP BO study.

Every plot_* function RETURNS a matplotlib Figure, so it shows inline in a
notebook (and you can also fig.savefig(...) it). Keeps analysis.ipynb clean.

    import plot_utils as pu
    runs = pu.load_runs()
    pu.plot_convergence(runs)        # displays inline

Run with the ml_gp_env Python (numpy / scipy / matplotlib).
"""
import os, glob, warnings
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))   # .../study/utils
STUDY = os.path.dirname(HERE)                         # .../study
RESULTS = os.path.join(STUDY, "results")
VAR_FCTR = np.array([15, 2, 8, 0, 10.])           # actual value of each level
NOISE_MULS = np.array([1.00, 0.70, 0.90, 0.50, 1.20]) * 10   # per-level noise mult.
LB, UB = -5, 10


# ============================ data / helpers ============================
def _s(x):
    a = np.ravel(x)
    return a[0] if a.size else None


def load_runs(results_dir=RESULTS):
    """Load every result .mat under results_dir into a list of dicts."""
    runs = []
    for f in sorted(glob.glob(os.path.join(results_dir, "**", "*.mat"), recursive=True)):
        m = scipy.io.loadmat(f)
        if "Y_min_history" not in m:
            continue
        meta = m["meta"][0, 0]
        runs.append(dict(
            acf=str(_s(meta["acf"])), param=float(_s(meta["acf_param"])),
            n_rep=int(_s(meta["n_rep"])), seed=int(_s(meta["seed"])),
            runtime=float(_s(meta["runtime"])),
            Y_min_history=np.ravel(m["Y_min_history"]).astype(float),
            X_sampled=np.atleast_2d(m["X_sampled"]).astype(float),
            n_initial=int(_s(m["n_initial"])),
            X_min_est=np.atleast_2d(m["X_min_est"]).astype(float),
            Y_sampled=np.ravel(m["Y_sampled"]).astype(float),
            X_best_final=np.ravel(m["X_best_final"]).astype(float),
            Y_best_final=float(_s(m["Y_best_final"])),
            Y_var_best_final=float(_s(m["Y_var_best_final"])),
        ))
    return runs


def f_true(x1, x2):
    return ((x2 - 5.1/(4*np.pi**2)*x1**2 + 5/np.pi*x1 - 6)**2
            + 10*(1 - 1/(8*np.pi))*np.cos(x1) + 10)


def sigma(x1, level):
    return 0.135*np.exp((0.15*x1)**2) * NOISE_MULS[level-1]


def ground_truth_min():
    x1 = np.linspace(LB, UB, 4000)
    return float(min(f_true(x1, v).min() for v in VAR_FCTR))


def true_opt_location():
    x1 = np.linspace(LB, UB, 4000); best = (np.inf, None, None)
    for lv, v in enumerate(VAR_FCTR, 1):
        fv = f_true(x1, v)
        if fv.min() < best[0]:
            best = (fv.min(), lv, x1[fv.argmin()])
    return best[1], best[2]          # level, x1


def label(acf, param):
    if acf == "haei":  return f"HAEI(γ={param:g})"
    if acf == "rahbo": return f"RAHBO(α={param:g})"
    if acf == "anpei": return f"ANPEI(β={param:g})"
    return acf.upper()


def cfg_key(r):
    p = r["param"]
    return (r["acf"], "na" if p != p else round(float(p), 6))


def config_colors(runs):
    cfgs = sorted({cfg_key(r) for r in runs}, key=lambda c: (c[0], str(c[1])))
    cmap = plt.get_cmap("tab10" if len(cfgs) <= 10 else "tab20")
    return {c: cmap(i % cmap.N) for i, c in enumerate(cfgs)}, cfgs


def true_min_per_category():
    """True noise-free minimum of f within each category level."""
    x1 = np.linspace(LB, UB, 4000)
    return np.array([f_true(x1, v).min() for v in VAR_FCTR])


def best_obj_per_category(run):
    """Lowest sample-mean objective found in each category level (NaN if none)."""
    X, Y = run["X_sampled"], run["Y_sampled"]
    lv = np.round(X[:, 1]).astype(int)
    out = np.full(len(VAR_FCTR), np.nan)
    for L in range(1, len(VAR_FCTR)+1):
        m = lv == L
        if m.any():
            out[L-1] = Y[m].min()
    return out


def summary(runs):
    """Quick text summary of what's loaded."""
    cfgs = sorted({cfg_key(r) for r in runs})
    print(f"{len(runs)} runs | {len(cfgs)} configs | "
          f"n_rep={sorted({r['n_rep'] for r in runs})} | "
          f"seeds={len({r['seed'] for r in runs})} | "
          f"ground-truth min = {ground_truth_min():.3f}")


# ============================ main plots ============================
def plot_convergence(runs, gt=None):
    """Best objective vs iteration, mean ± 95% CI per acquisition, faceted by n_rep."""
    if gt is None: gt = ground_truth_min()
    colors, cfgs = config_colors(runs)
    n_reps = sorted({r["n_rep"] for r in runs})
    fig, axes = plt.subplots(1, len(n_reps), figsize=(6*len(n_reps), 4.5),
                             squeeze=False, sharey=True)
    for ax, nr in zip(axes[0], n_reps):
        for cfg in cfgs:
            hist = [r["Y_min_history"] for r in runs if cfg_key(r) == cfg and r["n_rep"] == nr]
            if not hist: continue
            L = min(len(h) for h in hist); H = np.array([h[:L] for h in hist])
            mean = H.mean(0)
            sem = H.std(0, ddof=1)/np.sqrt(H.shape[0]) if H.shape[0] > 1 else np.zeros(L)
            it = np.arange(1, L+1)
            ax.plot(it, mean, color=colors[cfg], label=label(*cfg))
            ax.fill_between(it, mean-1.96*sem, mean+1.96*sem, color=colors[cfg], alpha=0.18)
        ax.axhline(gt, ls="--", color="k", lw=1.2, label="ground truth")
        ax.set_title(f"n_rep = {nr}"); ax.set_xlabel("Iteration"); ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("Best sample-mean objective $y^*$")
    axes[0][-1].legend(fontsize=8, loc="upper right")
    fig.suptitle("Best objective vs iteration (mean ± 95% CI)"); fig.tight_layout()
    return fig


def true_obj_at_recommended(run):
    """True (noise-free) objective at the model's recommended optimum (X_min_est),
    per iteration. Always >= the global min (it's f evaluated at a real point)."""
    out = []
    for xr in run["X_min_est"]:
        lv = int(round(xr[1]))
        out.append(float(f_true(np.array([xr[0]]), VAR_FCTR[lv-1])[0])
                   if 1 <= lv <= len(VAR_FCTR) else np.nan)
    return np.array(out)


def plot_convergence_true(runs, gt=None):
    """Convergence using the TRUE noise-free objective at the recommended optimum
    each iteration (X_min_est) instead of the optimistically-biased best noisy
    sample-mean. The curves sit ABOVE the true global minimum and show how good
    the recommended design actually is vs ground truth."""
    if gt is None: gt = ground_truth_min()
    colors, cfgs = config_colors(runs)
    n_reps = sorted({r["n_rep"] for r in runs})
    fig, axes = plt.subplots(1, len(n_reps), figsize=(6*len(n_reps), 4.5),
                             squeeze=False, sharey=True)
    for ax, nr in zip(axes[0], n_reps):
        for cfg in cfgs:
            curves = [true_obj_at_recommended(r) for r in runs
                      if cfg_key(r) == cfg and r["n_rep"] == nr]
            if not curves: continue
            L = min(len(c) for c in curves); C = np.array([c[:L] for c in curves])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mean = np.nanmean(C, 0)
                sem = (np.nanstd(C, 0, ddof=1)/np.sqrt(C.shape[0])) if C.shape[0] > 1 else np.zeros(L)
            sem = np.nan_to_num(sem)
            it = np.arange(1, L+1)
            ax.plot(it, mean, color=colors[cfg], label=label(*cfg))
            ax.fill_between(it, mean-1.96*sem, mean+1.96*sem, color=colors[cfg], alpha=0.18)
        ax.axhline(gt, ls="--", color="k", lw=1.2, label="true global min")
        ax.set_title(f"n_rep = {nr}"); ax.set_xlabel("Iteration"); ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("True objective at recommended optimum")
    axes[0][-1].legend(fontsize=8, loc="upper right")
    fig.suptitle("Convergence vs ground truth — true objective at recommended optimum (mean ± 95% CI)")
    fig.tight_layout()
    return fig


def _x1_curve(run, source):
    """x1 of the best location per iteration. source='recommended' -> the model's
    X_min_est; source='best' -> x1 of the best-by-sample-mean design so far."""
    if source == "best":
        X, Y, n0 = run["X_sampled"], run["Y_sampled"], run["n_initial"]
        return np.array([X[int(np.argmin(Y[:n0+i])), 0] for i in range(1, len(Y)-n0+1)])
    return run["X_min_est"][:, 0]


def _x1_final(run, source):
    return float(run["X_best_final"][0]) if source == "best" else float(run["X_min_est"][-1, 0])


def plot_x1_convergence(runs, source="recommended"):
    """x1 of the best location vs iteration (mean ± 95% CI), per acquisition,
    faceted by n_rep, vs the true optimal x1 (= 3.18, category 2 — where every
    best design lands). source: 'recommended' (X_min_est) or 'best' (best-by-
    sample-mean design, the optimistically-biased one)."""
    gt_lv, gt_x1 = true_opt_location()
    colors, cfgs = config_colors(runs)
    n_reps = sorted({r["n_rep"] for r in runs})
    fig, axes = plt.subplots(1, len(n_reps), figsize=(6*len(n_reps), 4.5),
                             squeeze=False, sharey=True)
    for ax, nr in zip(axes[0], n_reps):
        for cfg in cfgs:
            curves = [_x1_curve(r, source) for r in runs
                      if cfg_key(r) == cfg and r["n_rep"] == nr]
            if not curves: continue
            L = min(len(c) for c in curves); C = np.array([c[:L] for c in curves])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mean = np.nanmean(C, 0)
                sem = (np.nanstd(C, 0, ddof=1)/np.sqrt(C.shape[0])) if C.shape[0] > 1 else np.zeros(L)
            it = np.arange(1, L+1)
            ax.plot(it, mean, color=colors[cfg], label=label(*cfg))
            ax.fill_between(it, mean-1.96*np.nan_to_num(sem), mean+1.96*np.nan_to_num(sem),
                            color=colors[cfg], alpha=0.18)
        ax.axhline(gt_x1, ls="--", color="k", lw=1.4, label=f"true $x_1$ = {gt_x1:.2f}")
        ax.set_title(f"n_rep = {nr}"); ax.set_xlabel("Iteration"); ax.grid(alpha=0.3)
    axes[0][0].set_ylabel(f"$x_1$ of {source} best (category 2)")
    axes[0][-1].legend(fontsize=8, loc="upper right")
    fig.suptitle(f"Best $x_1$ vs iteration — convergence to true optimum $x_1$ ({source})")
    fig.tight_layout()
    return fig


def plot_x1_distribution(runs, n_rep=10, source="recommended", show_violin=True, show_mean=True):
    """Distribution of the FINAL best x1 across seeds, per acquisition, vs the true
    optimal x1. source: 'recommended' or 'best'. show_violin=False -> drop the
    shaded shape; show_mean=False -> drop the mean tick (misleading when bimodal)."""
    gt_lv, gt_x1 = true_opt_location()
    colors, cfgs = config_colors(runs)
    data, labs, cols = [], [], []
    for cfg in cfgs:
        x1s = [_x1_final(r, source) for r in runs
               if cfg_key(r) == cfg and r["n_rep"] == n_rep]
        if x1s:
            data.append(x1s); labs.append(label(*cfg)); cols.append(colors[cfg])
    fig, ax = plt.subplots(figsize=(1.05*len(data)+3, 5))
    pos = np.arange(1, len(data)+1)
    if show_violin:
        parts = ax.violinplot(data, positions=pos, showmeans=False, showextrema=False, widths=0.8)
        for pc, c in zip(parts["bodies"], cols):
            pc.set_facecolor(c); pc.set_alpha(0.35)
    rng = np.random.default_rng(0)
    for i, (d, c) in zip(pos, zip(data, cols)):
        ax.scatter(i + rng.uniform(-0.1, 0.1, len(d)), d, s=16, color=c,
                   alpha=0.7, edgecolor="k", linewidth=0.25, zorder=3)
        if show_mean:
            ax.plot([i-0.28, i+0.28], [np.mean(d)]*2, color="k", lw=2, zorder=4)   # mean tick
    ax.axhline(gt_x1, ls="--", color="k", lw=1.6, label=f"true $x_1$ = {gt_x1:.2f}")
    ax.set_xticks(pos); ax.set_xticklabels(labs, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel(f"final $x_1$ of {source} best")
    ax.set_title(f"Distribution of final best $x_1$ vs true optimum "
                 f"(n_rep={n_rep}, category 2, {source})")
    ax.grid(alpha=0.3, axis="y"); ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    return fig


def final_true_obj(run, source="recommended"):
    """True (noise-free) objective of the final best/recommended design."""
    xr = run["X_best_final"] if source == "best" else run["X_min_est"][-1]
    lv = int(round(xr[1]))
    return float(f_true(np.array([xr[0]]), VAR_FCTR[lv-1])[0]) if 1 <= lv <= len(VAR_FCTR) else np.nan


def plot_final_boxplots(runs, source="recommended", noisy=False):
    """Boxplot of the FINAL objective across seeds, per acquisition, faceted by
    n_rep, vs the true global min. noisy=True -> the biased best NOISY sample-mean
    (Y_best_final, can dip below the truth); noisy=False (default) -> the TRUE
    noise-free objective at the final recommended design (always >= the true min)."""
    gt = ground_truth_min()
    colors, cfgs = config_colors(runs)
    n_reps = sorted({r["n_rep"] for r in runs})
    fig, axes = plt.subplots(1, len(n_reps), figsize=(5.5*len(n_reps), 4.5),
                             squeeze=False, sharey=True)
    for ax, nr in zip(axes[0], n_reps):
        data, labs, cols = [], [], []
        for cfg in cfgs:
            if noisy:
                vals = [r["Y_best_final"] for r in runs
                        if cfg_key(r) == cfg and r["n_rep"] == nr]
            else:
                vals = [final_true_obj(r, source) for r in runs
                        if cfg_key(r) == cfg and r["n_rep"] == nr]
            if vals:
                data.append(vals); labs.append(label(*cfg)); cols.append(colors[cfg])
        bp = ax.boxplot(data, patch_artist=True, showmeans=True)
        for patch, c in zip(bp["boxes"], cols):
            patch.set_facecolor(c); patch.set_alpha(0.5)
        ax.axhline(gt, ls="--", color="k", lw=1.2, label="true global min")
        ax.set_xticklabels(labs, rotation=40, ha="right", fontsize=8)
        ax.set_title(f"n_rep = {nr}"); ax.grid(alpha=0.3, axis="y")
    kind = "noisy best sample-mean" if noisy else f"true objective at {source} design"
    axes[0][0].set_ylabel(f"final objective\n({kind})")
    axes[0][-1].legend(fontsize=8)
    fig.suptitle(f"Final-objective distribution across seeds — {'noisy' if noisy else 'noise-free'}")
    fig.tight_layout()
    return fig


def plot_initial_doe(runs, n_rep=None):
    """The initial DOE (the 15 LHS starting points = 3 per category, before BO)
    overlaid across all seeds, x1 vs category — shows the coverage/spread of the
    initial design. The DOE is identical across acquisitions for a given
    (seed, n_rep), so it is deduplicated. n_rep=None pools all n_rep settings."""
    gt_lv, gt_x1 = true_opt_location()
    seen = {}
    for r in runs:
        if n_rep is not None and r["n_rep"] != n_rep:
            continue
        seen.setdefault((r["seed"], r["n_rep"]), r["X_sampled"][:r["n_initial"]])
    if not seen:
        raise ValueError("no runs match")
    pts = np.vstack(list(seen.values()))
    lv = np.round(pts[:, 1]).astype(int); nlev = len(VAR_FCTR)
    fig, ax = plt.subplots(figsize=(8, 5))
    rng = np.random.default_rng(0)
    ax.scatter(lv + rng.uniform(-0.18, 0.18, len(lv)), pts[:, 0], s=16,
               c="steelblue", alpha=0.35, edgecolor="k", linewidth=0.2,
               label="initial DOE points")
    ax.plot(gt_lv, gt_x1, marker="*", ms=16, c="crimson", zorder=5, label="true optimum")
    ax.set_xticks(range(1, nlev+1))
    ax.set_xticklabels([f"{i+1}\n({v:g})" for i, v in enumerate(VAR_FCTR)])
    ax.set_xlabel("categorical level (value)"); ax.set_ylabel("$x_1$")
    ax.set_ylim(LB-0.5, UB+0.5)
    per_cat = len(pts) // (len(seen) * nlev)
    nrtxt = "all n_rep" if n_rep is None else f"n_rep={n_rep}"
    ax.set_title(f"Initial DOE across {len(seen)} designs ({nrtxt}) — "
                 f"{len(pts)} points ({per_cat} LHS per category)")
    ax.grid(alpha=0.3); ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_input_space(runs, n_rep=None):
    """x1 vs categorical level sampling, colored by BO iteration, per acquisition."""
    colors, cfgs = config_colors(runs)
    if n_rep is None: n_rep = max({r["n_rep"] for r in runs})
    cfgs = [c for c in cfgs if any(cfg_key(r) == c and r["n_rep"] == n_rep for r in runs)]
    nlev = len(VAR_FCTR); cols = min(4, len(cfgs)); rows = int(np.ceil(len(cfgs)/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3.2*rows),
                             squeeze=False, sharex=True, sharey=True)
    sc = None
    for ax, cfg in zip(axes.ravel(), cfgs):
        init_x, bo_x, bo_it = [], [], []
        for r in runs:
            if cfg_key(r) != cfg or r["n_rep"] != n_rep: continue
            X, n0 = r["X_sampled"], r["n_initial"]
            init_x.append(X[:n0]); bo_x.append(X[n0:]); bo_it.append(np.arange(1, len(X)-n0+1))
        if init_x:
            I = np.vstack(init_x); B = np.vstack(bo_x); T = np.concatenate(bo_it)
            ax.scatter(I[:, 1], I[:, 0], c="0.6", marker="x", s=25, label="initial DOE")
            sc = ax.scatter(B[:, 1], B[:, 0], c=T, cmap="viridis", s=22,
                            edgecolor="k", linewidth=0.3, label="BO samples")
        ax.set_title(label(*cfg), fontsize=9)
        ax.set_xticks(range(1, nlev+1))
        ax.set_xticklabels([f"{i+1}\n({v:g})" for i, v in enumerate(VAR_FCTR)], fontsize=7)
        ax.grid(alpha=0.3)
    for ax in axes[-1]: ax.set_xlabel("level idx (value)")
    for ax in axes[:, 0]: ax.set_ylabel("$x_1$")
    if sc is not None:
        fig.colorbar(sc, ax=axes.ravel().tolist(), label="BO iteration", shrink=0.6)
    fig.suptitle(f"Input-space sampling (n_rep={n_rep}, pooled over seeds)")
    return fig


def plot_level_histogram(runs):
    """Categorical-level selection frequency (BO samples), one panel per n_rep."""
    n_reps = sorted({r["n_rep"] for r in runs}); nlev = len(VAR_FCTR)
    fig, axes = plt.subplots(1, len(n_reps), figsize=(5*len(n_reps), 4),
                             squeeze=False, sharey=True)
    for ax, nr in zip(axes[0], n_reps):
        counts = np.zeros(nlev)
        for r in runs:
            if r["n_rep"] != nr: continue
            for lv in r["X_sampled"][r["n_initial"]:, 1].astype(int):
                if 1 <= lv <= nlev: counts[lv-1] += 1
        ax.bar(range(1, nlev+1), counts, color="steelblue", edgecolor="k")
        ax.set_title(f"n_rep = {nr}"); ax.set_xlabel("categorical level (value)")
        ax.set_xticks(range(1, nlev+1))
        ax.set_xticklabels([f"{i+1}\n({v:g})" for i, v in enumerate(VAR_FCTR)], fontsize=8)
        ax.grid(alpha=0.3, axis="y")
    axes[0][0].set_ylabel("# BO samples")
    fig.suptitle("Categorical level selection frequency"); fig.tight_layout()
    return fig


def plot_objective_variance(runs):
    """Final solutions in objective vs aleatoric-variance space."""
    colors, cfgs = config_colors(runs)
    fig, ax = plt.subplots(figsize=(7, 5.5)); markers = {3: "o", 5: "s", 10: "^"}
    for cfg in cfgs:
        for nr in sorted({r["n_rep"] for r in runs}):
            pts = [(r["Y_best_final"], r["Y_var_best_final"]) for r in runs
                   if cfg_key(r) == cfg and r["n_rep"] == nr]
            if not pts: continue
            pts = np.array(pts)
            ax.scatter(pts[:, 0], pts[:, 1], color=colors[cfg], marker=markers.get(nr, "o"),
                       s=40, alpha=0.8, edgecolor="k", linewidth=0.3)
    cfg_h = [Line2D([], [], color=colors[c], marker="o", ls="", label=label(*c)) for c in cfgs]
    nr_h = [Line2D([], [], color="0.4", marker=markers[k], ls="", label=f"n_rep={k}")
            for k in markers if any(r["n_rep"] == k for r in runs)]
    ax.add_artist(ax.legend(handles=cfg_h, title="acquisition", fontsize=8, loc="upper right"))
    ax.legend(handles=nr_h, title="replicates", fontsize=8, loc="lower right")
    ax.set_xlabel("Objective of final solution  $y$")
    ax.set_ylabel("Aleatoric variance of final solution")
    ax.set_title("Final solutions in objective–variance space"); ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_best_designs(runs, n_rep=10, expected_cfgs=None):
    """Best design per seed compared to true optimum, with placeholders for live data.

    expected_cfgs: optional list of (acf, param) tuples (e.g. ACQ_CONFIGS) so that
    not-yet-run configs appear as 'Awaiting Data' panels instead of vanishing."""
    colors, cfgs_in_runs = config_colors(runs)

    # Build the grid from expected configs if given; normalize to cfg_key format
    # so ei/lcb/pi (param=NaN -> "na") match correctly.
    if expected_cfgs is not None:
        all_cfgs = [(a, "na" if p != p else round(float(p), 6)) for (a, p) in expected_cfgs]
    else:
        all_cfgs = cfgs_in_runs
        
    gt_lv, gt_x1 = true_opt_location()
    nlev = len(VAR_FCTR)
    
    cols = min(4, len(all_cfgs))
    rows = int(np.ceil(len(all_cfgs)/cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(3.8*cols, 3.4*rows),
                             squeeze=False, sharex=True, sharey=True)
    rng = np.random.default_rng(0)
    
    # 1. Apply baseline formatting to EVERY subplot (including empty ones)
    for ax in axes.ravel():
        ax.set_xticks(range(1, nlev+1))
        ax.set_xticklabels([f"{i+1}\n({v:g})" for i, v in enumerate(VAR_FCTR)], fontsize=8)
        ax.set_xlim(0.5, nlev + 0.5)        # show all categories, not just level 2
        ax.grid(axis='y', alpha=0.4, linestyle='--')
        ax.grid(axis='x', alpha=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # 2. Populate the subplots with available data
    for i, cfg in enumerate(all_cfgs):
        ax = axes.ravel()[i]
        
        # Extract runs for this specific config
        B = np.array([r["X_best_final"] for r in runs
                      if cfg_key(r) == cfg and r["n_rep"] == n_rep])
        
        if len(B) > 0:
            lv = np.round(B[:, 1]).astype(int)
            jit = rng.uniform(-0.15, 0.15, size=len(lv))
            
            # Scatter points
            ax.scatter(lv + jit, B[:, 0], c=[colors.get(cfg, 'gray')], s=40, alpha=0.5,
                       edgecolor="white", linewidth=0.5)
            
            # True Optimum Marker (small, unobtrusive)
            ax.plot(gt_lv, gt_x1, marker="X", ms=5, c="black", alpha=0.7, zorder=5)
            
            frac = np.mean(lv == gt_lv) * 100
            ax.set_title(f"{label(*cfg)} — {frac:.0f}% optimal lvl", fontsize=10, weight='medium')
        else:
            # Handle pending experiments
            ax.set_title(f"{label(*cfg)}", fontsize=10, weight='medium', color='gray')
            ax.text(0.5, 0.5, "Awaiting Data...", ha='center', va='center', 
                    transform=ax.transAxes, color='gray', style='italic', fontsize=9)

    # 3. Clean up any completely excess axes (if grid size > len(all_cfgs))
    for i in range(len(all_cfgs), rows * cols):
        axes.ravel()[i].set_axis_off()

    # 4. Set Labels
    for ax in axes[-1]: 
        if ax.get_visible(): 
            ax.set_xlabel("categorical level (value)", fontsize=9)
            
    for ax in axes[:, 0]: 
        ax.set_ylabel("$x_1$", fontsize=10)
        
    # --- Create Custom Master Legend ---
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', alpha=0.6,
               markeredgecolor='white', markersize=10, label='Per-seed best'),
        Line2D([0], [0], marker='X', color='w', markerfacecolor='black', alpha=0.7,
               markersize=7, label='True optimum')
    ]
    
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, 
               frameon=False, fontsize=11, bbox_to_anchor=(0.5, 0.01))

    fig.suptitle(f"Best design per seed (lowest sample-mean location), n_rep={n_rep}", 
                 fontsize=12, weight='bold', y=0.98)
    
    fig.tight_layout(rect=[0, 0.08, 1, 0.96])
    
    return fig


def plot_best_per_category(runs, n_rep=10, expected_cfgs=None):
    """Best objective found in EACH category (mean ± 95% CI across seeds) vs the
    true per-category minimum, per acquisition. Shows ALL 5 categories (unlike
    plot_best_designs, where the final pick is almost always level 2).

    expected_cfgs: optional list of (acf, param) tuples (e.g. ACQ_CONFIGS) to also
    show every acquisition panel, with placeholders for not-yet-run ones."""
    base_colors, present = config_colors(runs)
    if expected_cfgs is not None:
        cfgs = [(a, "na" if p != p else round(float(p), 6)) for (a, p) in expected_cfgs]
        cmap = plt.get_cmap("tab20" if len(cfgs) > 10 else "tab10")
        colors = {c: cmap(i % cmap.N) for i, c in enumerate(cfgs)}
    else:
        cfgs = [c for c in present if any(cfg_key(r) == c and r["n_rep"] == n_rep for r in runs)]
        colors = base_colors
        if not cfgs:
            raise ValueError(f"no data at n_rep={n_rep}")
    tmin = true_min_per_category(); nlev = len(VAR_FCTR); x = np.arange(1, nlev+1)
    cols = min(3, len(cfgs)); rows = int(np.ceil(len(cfgs)/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2*cols, 3.1*rows),
                             squeeze=False, sharex=True, sharey=True)
    for ax, cfg in zip(axes.ravel(), cfgs):
        ax.set_xticks(x)
        ax.set_xticklabels([f"{i+1}\n({v:g})" for i, v in enumerate(VAR_FCTR)], fontsize=7)
        ax.grid(alpha=0.3)
        M = np.array([best_obj_per_category(r) for r in runs
                      if cfg_key(r) == cfg and r["n_rep"] == n_rep])          # [seeds, 5]
        if M.size == 0:                                  # not-yet-run config
            ax.set_title(f"{label(*cfg)}  (awaiting data)", fontsize=9, color="gray")
            ax.text(0.5, 0.5, "Awaiting Data...", ha="center", va="center",
                    transform=ax.transAxes, color="gray", style="italic", fontsize=8)
            continue
        cnt = np.sum(~np.isnan(M), 0)
        with warnings.catch_warnings():                  # quiet n<2 std warnings
            warnings.simplefilter("ignore")
            mean = np.nanmean(M, 0)
            sem = np.nan_to_num(np.nanstd(M, 0, ddof=1) / np.sqrt(np.maximum(cnt, 1)))
        ax.plot(x, tmin, "kX--", ms=10, lw=1, label="true min / category", zorder=2)
        ax.errorbar(x, mean, yerr=1.96*sem, marker="o", ms=6, color=colors[cfg],
                    capsize=3, lw=1.5, label="best found (mean ± 95% CI)", zorder=3)
        ax.set_title(f"{label(*cfg)}  (n={M.shape[0]} seeds)", fontsize=9)
        ax.legend(fontsize=6.5, loc="upper left")
    for ax in axes[-1]: ax.set_xlabel("categorical level (value)")
    for ax in axes[:, 0]: ax.set_ylabel("best objective in category")
    for ax in axes.ravel()[len(cfgs):]: ax.set_visible(False)     # hide excess panels
    fig.suptitle(f"Best objective found within each category, n_rep={n_rep}")
    fig.tight_layout()
    return fig


def plot_single_run(mat_path):
    """Per-category slices of ONE run: true f + noise band + where BO sampled."""
    if not os.path.isabs(mat_path):
        mat_path = os.path.join(STUDY, mat_path) if mat_path.startswith("results") \
            else os.path.join(RESULTS, mat_path)
    m = scipy.io.loadmat(mat_path)
    X = np.atleast_2d(m["X_sampled"]).astype(float); Y = np.ravel(m["Y_sampled"]).astype(float)
    n0 = int(_s(m["n_initial"])); Xme = np.atleast_2d(m["X_min_est"]).astype(float)
    meta = m["meta"][0, 0]
    acf = str(_s(meta["acf"])); param = float(_s(meta["acf_param"]))
    nrep = int(_s(meta["n_rep"])); seed = int(_s(meta["seed"]))
    gt_lv, gt_x1 = true_opt_location(); x1g = np.linspace(LB, UB, 400)
    levels = np.round(X[:, 1]).astype(int); order = np.arange(len(X)); nlev = len(VAR_FCTR)
    fig, axes = plt.subplots(1, nlev, figsize=(3.6*nlev, 4.2), sharey=True)
    for lv in range(1, nlev+1):
        ax = axes[lv-1]; ft = f_true(x1g, VAR_FCTR[lv-1]); s = sigma(x1g, lv)
        ax.fill_between(x1g, ft-1.96*s, ft+1.96*s, color="0.85", label="true ±1.96σ")
        ax.plot(x1g, ft, "k-", lw=1.5, label="true f")
        sel = levels == lv; init = sel & (order < n0); bo = sel & (order >= n0)
        ax.scatter(X[init, 0], Y[init], c="0.5", marker="x", s=45, label="initial DOE")
        if bo.any():
            it = order[bo] - n0 + 1
            ax.scatter(X[bo, 0], Y[bo], c=it, cmap="viridis", s=42,
                       edgecolor="k", linewidth=0.4, zorder=3, label="BO samples")
        if int(round(Xme[-1, 1])) == lv:
            ax.axvline(Xme[-1, 0], color="r", ls=":", lw=1.6, label="recommended x*")
        if lv == gt_lv:
            ax.plot(gt_x1, f_true(np.array([gt_x1]), VAR_FCTR[lv-1])[0], "m*", ms=15,
                    zorder=4, label="global optimum")
        ax.set_title(f"level {lv} (val {VAR_FCTR[lv-1]:g}, noise×{NOISE_MULS[lv-1]:g})", fontsize=9)
        ax.set_xlabel("$x_1$"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("objective $y$")
    plabel = "" if param != param else f"={param:g}"
    fig.suptitle(f"Single BO run — {acf}{plabel}, n_rep={nrep}, seed={seed}", y=1.02)
    fig.tight_layout()
    return fig


# ============================ live progress ============================
# Canonical 12-config sweep; folder tag matches run_sweep.acf_tag().
_KNOB = {"haei": "g", "rahbo": "a", "anpei": "b"}
SWEEP_CONFIGS = [
    ("lcb", float("nan")), ("pi", float("nan")), ("ei", float("nan")),
    ("haei", 0.5), ("haei", 1.0), ("haei", 5.0),
    ("anpei", 0.2), ("anpei", 0.5), ("anpei", 0.8),
    ("rahbo", 0.5), ("rahbo", 1.0), ("rahbo", 5.0),
]


def _acf_tag(acf, param):
    if param != param:                       # NaN -> no knob (ei/lcb/pi)
        return acf
    return f"{acf}_{_KNOB.get(acf, 'p')}{param:g}"


def plot_progress(target_per_acq=90, results_dir=RESULTS):
    """Live sweep progress: one bar per acquisition (count of completed .mat),
    with a dashed target line at `target_per_acq` (3 n_rep x 30 seeds = 90).
    Reads the results directory directly, so it works while the sweep runs."""
    labels, counts = [], []
    for acf, p in SWEEP_CONFIGS:
        d = os.path.join(results_dir, _acf_tag(acf, p))
        counts.append(len(glob.glob(os.path.join(d, "**", "*.mat"), recursive=True)))
        labels.append(label(acf, p))
    counts = np.array(counts)
    done, total = int(counts.sum()), target_per_acq * len(SWEEP_CONFIGS)

    fig, ax = plt.subplots(figsize=(11, 5))
    colors = ["#2e7d32" if c >= target_per_acq else "#4a90d9" for c in counts]
    bars = ax.bar(range(len(counts)), counts, color=colors, edgecolor="k", lw=0.4)
    ax.axhline(target_per_acq, color="crimson", ls="--", lw=1.5,
               label=f"target = {target_per_acq}")
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width()/2, c + target_per_acq*0.015, str(int(c)),
                ha="center", va="bottom", fontsize=8)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_ylim(0, target_per_acq * 1.12)
    ax.set_ylabel("completed runs")
    pct = 100 * done / total if total else 0
    ax.set_title(f"Sweep progress — {done}/{total} runs ({pct:.0f}%)  "
                 f"·  {int((counts >= target_per_acq).sum())}/{len(counts)} configs complete")
    ax.legend(loc="lower right"); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig
