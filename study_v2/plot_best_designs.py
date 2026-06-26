#!/usr/bin/env python
"""
Plot 2 variant: instead of all sampled iterations, show only the BEST design
(lowest sample-mean location, X_best_final) that each seed ended at — one marker
per seed — per acquisition, with the across-seed MEAN highlighted.

Axes: categorical level (x) vs x1 (y), matching plot 2's orientation.
Saved to plots/main/2b_best_designs_nrep<NN>.png   (default n_rep = 10)

Usage:  python plot_best_designs.py            # n_rep=10
        python plot_best_designs.py 3          # n_rep=3
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import analyze as A

VAR_FCTR = np.array([15, 2, 8, 0, 10.])


def f_true(x1, x2):
    return ((x2 - 5.1/(4*np.pi**2)*x1**2 + 5/np.pi*x1 - 6)**2
            + 10*(1 - 1/(8*np.pi))*np.cos(x1) + 10)


def true_opt():
    x1 = np.linspace(-5, 10, 4000); best = (np.inf, None, None)
    for lv, v in enumerate(VAR_FCTR, 1):
        fv = f_true(x1, v)
        if fv.min() < best[0]:
            best = (fv.min(), lv, x1[fv.argmin()])
    return best[1], best[2]      # level, x1


def main():
    n_rep = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    runs = A.load_runs()
    colors, cfgs = A.config_colors(runs)
    cfgs = [c for c in cfgs if any(A.cfg_key(r) == c and r["n_rep"] == n_rep for r in runs)]
    if not cfgs:
        print(f"no data at n_rep={n_rep}"); return
    gt_lv, gt_x1 = true_opt()
    nlev = len(VAR_FCTR)

    cols = min(4, len(cfgs)); rows = int(np.ceil(len(cfgs)/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6*cols, 3.2*rows),
                             squeeze=False, sharex=True, sharey=True)
    rng = np.random.default_rng(0)
    for ax, cfg in zip(axes.ravel(), cfgs):
        B = np.array([r["X_best_final"] for r in runs
                      if A.cfg_key(r) == cfg and r["n_rep"] == n_rep])  # [seeds, (x1, level)]
        lv = np.round(B[:, 1]).astype(int)
        jit = rng.uniform(-0.18, 0.18, size=len(lv))            # spread overlapping points
        ax.scatter(lv + jit, B[:, 0], c=[colors[cfg]], s=34, alpha=0.6,
                   edgecolor="k", linewidth=0.3, label=f"per-seed best (n={len(B)})")
        # mean best: modal level, mean x1 among seeds at that level
        modal = np.bincount(lv, minlength=nlev+1).argmax()
        mx1 = B[lv == modal, 0].mean()
        ax.scatter([modal], [mx1], marker="*", s=320, c=[colors[cfg]],
                   edgecolor="k", linewidth=0.8, zorder=4,
                   label=f"mean best (lvl {modal}, x1={mx1:.2f})")
        ax.plot(gt_lv, gt_x1, marker="X", ms=12, c="k", zorder=5, label="true optimum")
        # fraction of seeds that found the optimal level
        frac = np.mean(lv == gt_lv) * 100
        ax.set_title(f"{A.label(*cfg)}  —  {frac:.0f}% at optimal level", fontsize=9)
        ax.set_xticks(range(1, nlev+1))
        ax.set_xticklabels([f"{i+1}\n({v:g})" for i, v in enumerate(VAR_FCTR)], fontsize=7)
        ax.grid(alpha=0.3); ax.legend(fontsize=6.5, loc="upper right", framealpha=0.85)
    for ax in axes[-1]:
        ax.set_xlabel("categorical level (value)")
    for ax in axes[:, 0]:
        ax.set_ylabel("$x_1$")
    fig.suptitle(f"Best design per seed (lowest sample-mean location), n_rep={n_rep}", y=1.0)
    out = os.path.join(A.HERE, "plots", "main", f"2b_best_designs_nrep{n_rep:02d}.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
