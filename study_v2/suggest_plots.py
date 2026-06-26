#!/usr/bin/env python
"""
Suggested extra visualizations for the acquisition-function study.
Writes to study/plot_suggestions/ (separate from the main 4 in study/plots/).

Reuses the loader/helpers in analyze.py. Re-runnable on partial data; nothing
here re-runs the optimization.

Gallery:
  A  final-objective boxplots per acquisition (faceted by n_rep) + ground truth
  B  simple regret |best - truth| vs iteration, log scale (convergence rate)
  C  aleatoric variance of the incumbent best vs iteration (robustness)
  D  objective-variance Pareto front of final solutions
  E  n_rep sensitivity: final objective vs n_rep, line per acquisition
  F  summary heatmaps: mean final objective & mean final variance over the grid
  G  mean runtime per acquisition
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import analyze as A   # load_runs, cfg_key, label, ground_truth_min, config_colors

OUT = os.path.join(A.HERE, "plots", "suggestions")
os.makedirs(OUT, exist_ok=True)


def _save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=140, bbox_inches="tight")
    plt.close(fig)


def A_final_boxplots(runs, colors, cfgs, gt):
    n_reps = sorted({r["n_rep"] for r in runs})
    fig, axes = plt.subplots(1, len(n_reps), figsize=(5.5*len(n_reps), 4.5),
                             squeeze=False, sharey=True)
    for ax, nr in zip(axes[0], n_reps):
        data, labs, cols = [], [], []
        for cfg in cfgs:
            vals = [r["Y_best_final"] for r in runs
                    if A.cfg_key(r) == cfg and r["n_rep"] == nr]
            if vals:
                data.append(vals); labs.append(A.label(*cfg)); cols.append(colors[cfg])
        bp = ax.boxplot(data, patch_artist=True, showmeans=True)
        for patch, c in zip(bp["boxes"], cols):
            patch.set_facecolor(c); patch.set_alpha(0.5)
        ax.axhline(gt, ls="--", color="k", lw=1.2, label="ground truth")
        ax.set_xticklabels(labs, rotation=40, ha="right", fontsize=8)
        ax.set_title(f"n_rep = {nr}"); ax.grid(alpha=0.3, axis="y")
    axes[0][0].set_ylabel("Final best objective")
    axes[0][-1].legend(fontsize=8)
    fig.suptitle("A · Final-objective distribution across seeds")
    _save(fig, "A_final_boxplots.png")


def B_simple_regret(runs, colors, cfgs, gt):
    n_reps = sorted({r["n_rep"] for r in runs})
    fig, axes = plt.subplots(1, len(n_reps), figsize=(6*len(n_reps), 4.5),
                             squeeze=False, sharey=True)
    for ax, nr in zip(axes[0], n_reps):
        for cfg in cfgs:
            hist = [r["Y_min_history"] for r in runs
                    if A.cfg_key(r) == cfg and r["n_rep"] == nr]
            if not hist:
                continue
            L = min(len(h) for h in hist)
            H = np.array([h[:L] for h in hist])
            regret = np.maximum(np.abs(H - gt), 1e-6).mean(0)
            ax.semilogy(np.arange(1, L+1), regret, color=colors[cfg], label=A.label(*cfg))
        ax.set_title(f"n_rep = {nr}"); ax.set_xlabel("Iteration"); ax.grid(alpha=0.3, which="both")
    axes[0][0].set_ylabel("mean |best − truth|  (log)")
    axes[0][-1].legend(fontsize=8)
    fig.suptitle("B · Simple regret vs iteration (convergence rate)")
    _save(fig, "B_simple_regret.png")


def C_incumbent_variance(runs, colors, cfgs):
    """Aleatoric variance of the best-so-far design vs iteration."""
    n_reps = sorted({r["n_rep"] for r in runs})
    fig, axes = plt.subplots(1, len(n_reps), figsize=(6*len(n_reps), 4.5),
                             squeeze=False, sharey=True)
    for ax, nr in zip(axes[0], n_reps):
        for cfg in cfgs:
            curves = []
            for r in runs:
                if A.cfg_key(r) != cfg or r["n_rep"] != nr:
                    continue
                y, v, n0 = r["Y_sampled"], r["Y_var_sampled"], r["n_initial"]
                niter = len(y) - n0
                cur = []
                for i in range(1, niter+1):
                    k = n0 + i
                    inc = int(np.argmin(y[:k]))     # incumbent index so far
                    cur.append(v[inc])
                curves.append(cur)
            if not curves:
                continue
            L = min(len(c) for c in curves)
            C = np.array([c[:L] for c in curves])
            ax.plot(np.arange(1, L+1), C.mean(0), color=colors[cfg], label=A.label(*cfg))
        ax.set_title(f"n_rep = {nr}"); ax.set_xlabel("Iteration"); ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("aleatoric variance of incumbent")
    axes[0][-1].legend(fontsize=8)
    fig.suptitle("C · Robustness: noise at the current best design vs iteration")
    _save(fig, "C_incumbent_variance.png")


def D_pareto(runs, colors, cfgs):
    pts = np.array([[r["Y_best_final"], r["Y_var_best_final"]] for r in runs])
    keys = [A.cfg_key(r) for r in runs]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for cfg in cfgs:
        P = np.array([pts[i] for i in range(len(pts)) if keys[i] == cfg])
        ax.scatter(P[:, 0], P[:, 1], color=colors[cfg], s=35, alpha=0.75,
                   edgecolor="k", linewidth=0.3, label=A.label(*cfg))
    # Pareto frontier (minimize objective AND variance)
    order = np.argsort(pts[:, 0])
    front, best_v = [], np.inf
    for i in order:
        if pts[i, 1] <= best_v:
            front.append(pts[i]); best_v = pts[i, 1]
    front = np.array(front)
    ax.plot(front[:, 0], front[:, 1], "k--", lw=1.3, label="Pareto front")
    ax.scatter(front[:, 0], front[:, 1], facecolors="none", edgecolors="k", s=110, lw=1.3)
    ax.set_xlabel("Final objective  $y$"); ax.set_ylabel("Final aleatoric variance")
    ax.set_title("D · Objective–variance Pareto front of final solutions")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    _save(fig, "D_pareto_obj_var.png")


def E_nrep_sensitivity(runs, colors, cfgs, gt):
    n_reps = sorted({r["n_rep"] for r in runs})
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for cfg in cfgs:
        xs, ys, es = [], [], []
        for nr in n_reps:
            vals = [r["Y_best_final"] for r in runs
                    if A.cfg_key(r) == cfg and r["n_rep"] == nr]
            if vals:
                xs.append(nr); ys.append(np.mean(vals))
                es.append(1.96*np.std(vals, ddof=1)/np.sqrt(len(vals)) if len(vals) > 1 else 0)
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, color=colors[cfg], label=A.label(*cfg))
    ax.axhline(gt, ls="--", color="k", lw=1.2, label="ground truth")
    ax.set_xticks(n_reps); ax.set_xlabel("n_rep (replicates per location)")
    ax.set_ylabel("Final best objective (mean ± 95% CI)")
    ax.set_title("E · Sensitivity to experimental budget (n_rep)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    _save(fig, "E_nrep_sensitivity.png")


def F_heatmaps(runs, cfgs):
    n_reps = sorted({r["n_rep"] for r in runs})
    labs = [A.label(*c) for c in cfgs]
    Mobj = np.full((len(cfgs), len(n_reps)), np.nan)
    Mvar = np.full((len(cfgs), len(n_reps)), np.nan)
    for i, cfg in enumerate(cfgs):
        for j, nr in enumerate(n_reps):
            o = [r["Y_best_final"] for r in runs if A.cfg_key(r) == cfg and r["n_rep"] == nr]
            v = [r["Y_var_best_final"] for r in runs if A.cfg_key(r) == cfg and r["n_rep"] == nr]
            if o: Mobj[i, j] = np.mean(o)
            if v: Mvar[i, j] = np.mean(v)
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.5*len(cfgs)+3))
    for ax, M, title, cmap in [(axes[0], Mobj, "mean final objective", "viridis_r"),
                               (axes[1], Mvar, "mean final variance", "magma_r")]:
        im = ax.imshow(M, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(n_reps))); ax.set_xticklabels([f"n_rep={n}" for n in n_reps])
        ax.set_yticks(range(len(cfgs))); ax.set_yticklabels(labs, fontsize=8)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i,j]:.2g}", ha="center", va="center", fontsize=7,
                            color="w")
        ax.set_title(title); fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("F · Grid summary (rows = acquisition, cols = n_rep)")
    _save(fig, "F_summary_heatmaps.png")


def G_runtime(runs, colors, cfgs):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labs, means, cols = [], [], []
    for cfg in cfgs:
        rt = [r["runtime"] for r in runs if A.cfg_key(r) == cfg]
        if rt:
            labs.append(A.label(*cfg)); means.append(np.mean(rt)); cols.append(colors[cfg])
    ax.bar(range(len(labs)), means, color=cols, edgecolor="k")
    ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("mean runtime per run (s)"); ax.grid(alpha=0.3, axis="y")
    ax.set_title("G · Computational cost per acquisition")
    _save(fig, "G_runtime.png")


def main():
    runs = A.load_runs()
    if not runs:
        print("No results found."); return
    gt = A.ground_truth_min(runs[0]["var_fctr"])
    colors, cfgs = A.config_colors(runs)
    A_final_boxplots(runs, colors, cfgs, gt)
    B_simple_regret(runs, colors, cfgs, gt)
    C_incumbent_variance(runs, colors, cfgs)
    D_pareto(runs, colors, cfgs)
    E_nrep_sensitivity(runs, colors, cfgs, gt)
    F_heatmaps(runs, cfgs)
    G_runtime(runs, colors, cfgs)
    print(f"Wrote 7 suggestion plots to {OUT}")


if __name__ == "__main__":
    main()
