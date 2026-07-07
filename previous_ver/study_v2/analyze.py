#!/usr/bin/env python
"""
Build the study plots from study/results/*.mat (saved by study_driver.m).

Plots:
  1. Best objective vs iteration (mean +/- 95% CI), with ground-truth line.
  2. x1 vs x2 (categorical level) sample distribution in the input space.
  3. Histogram of sampled categorical level (x2), one panel per n_rep.
  4. Aleatoric variance vs objective for the final solutions.

Re-runnable any time; nothing here re-runs the optimization.
"""
import os, glob
import numpy as np
import scipy.io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
PLOTS = os.path.join(HERE, "plots", "main")
os.makedirs(PLOTS, exist_ok=True)


def _s(x):
    """Unwrap a scipy.io scalar/string."""
    a = np.ravel(x)
    return a[0] if a.size else None


def load_runs():
    runs = []
    for f in sorted(glob.glob(os.path.join(RESULTS, "**", "*.mat"), recursive=True)):
        m = scipy.io.loadmat(f)
        if "Y_min_history" not in m:
            continue
        meta = m["meta"][0, 0]
        acf = str(_s(meta["acf"]))
        param = float(_s(meta["acf_param"]))
        runs.append(dict(
            acf=acf, param=param,
            n_rep=int(_s(meta["n_rep"])), seed=int(_s(meta["seed"])),
            runtime=float(_s(meta["runtime"])),
            Y_min_history=np.ravel(m["Y_min_history"]).astype(float),
            X_sampled=np.atleast_2d(m["X_sampled"]).astype(float),
            Y_sampled=np.ravel(m["Y_sampled"]).astype(float),
            Y_var_sampled=np.ravel(m["Y_var_sampled"]).astype(float),
            n_initial=int(_s(m["n_initial"])),
            X_best_final=np.ravel(m["X_best_final"]).astype(float),
            Y_best_final=float(_s(m["Y_best_final"])),
            Y_var_best_final=float(_s(m["Y_var_best_final"])),
            var_fctr=np.ravel(m["var_fctr"]).astype(float),
        ))
    return runs


def label(acf, param):
    if acf == "haei":  return f"HAEI(γ={param:g})"
    if acf == "rahbo": return f"RAHBO(α={param:g})"
    if acf == "anpei": return f"ANPEI(β={param:g})"
    return acf.upper()


def ground_truth_min(var_fctr):
    """True noise-free global min of the test fn over x1 in [-5,10], x2 in var_fctr."""
    x1 = np.linspace(-5, 10, 4000)
    best = np.inf
    for x2 in var_fctr:
        f = (x2 - 5.1/(4*np.pi**2)*x1**2 + 5/np.pi*x1 - 6)**2 \
            + 10*(1 - 1/(8*np.pi))*np.cos(x1) + 10
        best = min(best, f.min())
    return best


def cfg_key(r):
    p = r["param"]
    return (r["acf"], "na" if p != p else round(float(p), 6))   # canonicalize NaN


# ----- color map per acf config -----
def config_colors(runs):
    cfgs = sorted({cfg_key(r) for r in runs}, key=lambda c: (c[0], c[1]))
    cmap = plt.get_cmap("tab10" if len(cfgs) <= 10 else "tab20")
    return {c: cmap(i % cmap.N) for i, c in enumerate(cfgs)}, cfgs


def plot1_convergence(runs, colors, cfgs, gt):
    n_reps = sorted({r["n_rep"] for r in runs})
    fig, axes = plt.subplots(1, len(n_reps), figsize=(6*len(n_reps), 4.5),
                             squeeze=False, sharey=True)
    for ax, nr in zip(axes[0], n_reps):
        for cfg in cfgs:
            hist = [r["Y_min_history"] for r in runs
                    if cfg_key(r) == cfg and r["n_rep"] == nr]
            if not hist:
                continue
            L = min(len(h) for h in hist)
            H = np.array([h[:L] for h in hist])          # seeds x iters
            mean = H.mean(0)
            sem = H.std(0, ddof=1) / np.sqrt(H.shape[0]) if H.shape[0] > 1 else np.zeros(L)
            it = np.arange(1, L+1)
            ax.plot(it, mean, color=colors[cfg], label=label(*cfg))
            ax.fill_between(it, mean-1.96*sem, mean+1.96*sem,
                            color=colors[cfg], alpha=0.18)
        ax.axhline(gt, ls="--", color="k", lw=1.2, label="ground truth")
        ax.set_title(f"n_rep = {nr}")
        ax.set_xlabel("Iteration"); ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("Best sample-mean objective $y^*$")
    axes[0][-1].legend(fontsize=8, loc="upper right")
    fig.suptitle("Best objective vs iteration (mean ± 95% CI)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "1_convergence.png"), dpi=140)
    plt.close(fig)


def plot2_input_space(runs, colors, cfgs):
    nr = max({r["n_rep"] for r in runs})   # show the richest replicate setting
    var_fctr = runs[0]["var_fctr"]
    ncfg = len(cfgs)
    cols = min(4, ncfg); rows = int(np.ceil(ncfg/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3.2*rows),
                             squeeze=False, sharex=True, sharey=True)
    for ax, cfg in zip(axes.ravel(), cfgs):
        init_x, bo_x, bo_it = [], [], []
        for r in runs:
            if cfg_key(r) != cfg or r["n_rep"] != nr:
                continue
            X, n0 = r["X_sampled"], r["n_initial"]
            init_x.append(X[:n0]); bo_x.append(X[n0:])
            bo_it.append(np.arange(1, len(X)-n0+1))
        if init_x:
            I = np.vstack(init_x); B = np.vstack(bo_x); T = np.concatenate(bo_it)
            # x2 (categorical level) on the horizontal axis, x1 on the vertical
            ax.scatter(I[:,1], I[:,0], c="0.6", marker="x", s=25, label="initial DOE")
            sc = ax.scatter(B[:,1], B[:,0], c=T, cmap="viridis", s=22,
                            edgecolor="k", linewidth=0.3, label="BO samples")
        ax.set_title(label(*cfg), fontsize=9)
        ax.set_xticks(range(1, len(var_fctr)+1))
        ax.set_xticklabels([f"{i+1}\n({v:g})" for i, v in enumerate(var_fctr)], fontsize=7)
        ax.grid(alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("level idx (value)")
    for ax in axes[:,0]:
        ax.set_ylabel("$x_1$")
    if init_x:
        fig.colorbar(sc, ax=axes.ravel().tolist(), label="BO iteration", shrink=0.6)
    fig.suptitle(f"Input-space sampling (n_rep={nr}, pooled over seeds)")
    fig.savefig(os.path.join(PLOTS, "2_input_space.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot3_level_hist(runs):
    n_reps = sorted({r["n_rep"] for r in runs})
    var_fctr = runs[0]["var_fctr"]
    nlev = len(var_fctr)
    fig, axes = plt.subplots(1, len(n_reps), figsize=(5*len(n_reps), 4),
                             squeeze=False, sharey=True)
    for ax, nr in zip(axes[0], n_reps):
        counts = np.zeros(nlev)
        for r in runs:
            if r["n_rep"] != nr:
                continue
            bo_levels = r["X_sampled"][r["n_initial"]:, 1].astype(int)
            for lv in bo_levels:
                if 1 <= lv <= nlev:
                    counts[lv-1] += 1
        ax.bar(range(1, nlev+1), counts, color="steelblue", edgecolor="k")
        ax.set_title(f"n_rep = {nr}")
        ax.set_xlabel("categorical level (value)")
        ax.set_xticks(range(1, nlev+1))
        ax.set_xticklabels([f"{i+1}\n({v:g})" for i, v in enumerate(var_fctr)], fontsize=8)
        ax.grid(alpha=0.3, axis="y")
    axes[0][0].set_ylabel("# BO samples")
    fig.suptitle("Categorical level selection frequency (BO samples, pooled over configs+seeds)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "3_level_histogram.png"), dpi=140)
    plt.close(fig)


def plot4_obj_var(runs, colors, cfgs):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    markers = {3: "o", 5: "s", 10: "^"}
    for cfg in cfgs:
        for nr in sorted({r["n_rep"] for r in runs}):
            pts = [(r["Y_best_final"], r["Y_var_best_final"])
                   for r in runs if cfg_key(r) == cfg and r["n_rep"] == nr]
            if not pts:
                continue
            pts = np.array(pts)
            ax.scatter(pts[:,0], pts[:,1], color=colors[cfg],
                       marker=markers.get(nr, "o"), s=40, alpha=0.8,
                       edgecolor="k", linewidth=0.3)
    # legends: color=config, marker=n_rep
    from matplotlib.lines import Line2D
    cfg_handles = [Line2D([], [], color=colors[c], marker="o", ls="", label=label(*c)) for c in cfgs]
    nr_handles = [Line2D([], [], color="0.4", marker=markers[k], ls="", label=f"n_rep={k}")
                  for k in markers if any(r["n_rep"] == k for r in runs)]
    leg1 = ax.legend(handles=cfg_handles, title="acquisition", fontsize=8, loc="upper right")
    ax.add_artist(leg1)
    ax.legend(handles=nr_handles, title="replicates", fontsize=8, loc="lower right")
    ax.set_xlabel("Objective of final solution  $y$")
    ax.set_ylabel("Aleatoric variance of final solution")
    ax.set_title("Final solutions in objective–variance space")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "4_objective_variance.png"), dpi=140)
    plt.close(fig)


def main():
    runs = load_runs()
    if not runs:
        print("No results found in", RESULTS)
        return
    print(f"Loaded {len(runs)} runs "
          f"({len({cfg_key(r) for r in runs})} configs, "
          f"n_rep={sorted({r['n_rep'] for r in runs})}, "
          f"{len({r['seed'] for r in runs})} seeds)")
    gt = ground_truth_min(runs[0]["var_fctr"])
    print(f"Ground-truth global min = {gt:.6g}")
    colors, cfgs = config_colors(runs)
    plot1_convergence(runs, colors, cfgs, gt)
    plot2_input_space(runs, colors, cfgs)
    plot3_level_hist(runs)
    plot4_obj_var(runs, colors, cfgs)
    print("Wrote plots to", PLOTS)


if __name__ == "__main__":
    main()
