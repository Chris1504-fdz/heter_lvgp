#!/usr/bin/env python
"""
Visualize a SINGLE BO run as per-category slices of the 2-D input space
(x1 continuous on the x-axis; one panel per categorical level x2).

Each panel shows: the true noise-free objective, the true heteroscedastic
noise band (+/-1.96 sigma), the initial DOE points, the BO-sampled points
(coloured by iteration), the recommended optimum, and the global optimum.

Usage:  python plot_single_run.py results/haei_g1/nrep10/seed01.mat
"""
import sys, os
import numpy as np
import scipy.io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VAR_FCTR = np.array([15, 2, 8, 0, 10.])           # actual value of each level
NOISE_MULS = np.array([1.00, 0.70, 0.90, 0.50, 1.20]) * 10   # per-level noise multiplier
LB, UB = -5, 10


def f_true(x1, x2):
    return ((x2 - 5.1/(4*np.pi**2)*x1**2 + 5/np.pi*x1 - 6)**2
            + 10*(1 - 1/(8*np.pi))*np.cos(x1) + 10)


def sigma(x1, level):                              # heteroscedastic noise std
    return 0.135*np.exp((0.15*x1)**2) * NOISE_MULS[level-1]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results/haei_g1/nrep10/seed01.mat"
    m = scipy.io.loadmat(path)
    X = np.atleast_2d(m["X_sampled"]).astype(float)      # [x1, level_idx]
    Y = np.ravel(m["Y_sampled"]).astype(float)
    n0 = int(np.ravel(m["n_initial"])[0])
    Xme = np.atleast_2d(m["X_min_est"]).astype(float)    # recommended optimum / iter
    meta = m["meta"][0, 0]
    acf = str(meta["acf"][0]); param = float(np.ravel(meta["acf_param"])[0])
    nrep = int(np.ravel(meta["n_rep"])[0]); seed = int(np.ravel(meta["seed"])[0])

    # global optimum (over the grid) for reference
    x1g = np.linspace(LB, UB, 600)
    gmin, gx1, glv = np.inf, None, None
    for lv in range(1, len(VAR_FCTR)+1):
        fv = f_true(x1g, VAR_FCTR[lv-1])
        if fv.min() < gmin:
            gmin, gx1, glv = fv.min(), x1g[fv.argmin()], lv

    levels = np.round(X[:, 1]).astype(int)
    order = np.arange(len(X))
    nlev = len(VAR_FCTR)
    fig, axes = plt.subplots(1, nlev, figsize=(3.6*nlev, 4.2), sharey=True)
    for lv in range(1, nlev+1):
        ax = axes[lv-1]
        ft = f_true(x1g, VAR_FCTR[lv-1]); s = sigma(x1g, lv)
        ax.fill_between(x1g, ft-1.96*s, ft+1.96*s, color="0.85",
                        label="true ±1.96σ noise")
        ax.plot(x1g, ft, "k-", lw=1.5, label="true f")
        sel = levels == lv
        init = sel & (order < n0)
        bo = sel & (order >= n0)
        ax.scatter(X[init, 0], Y[init], c="0.5", marker="x", s=45, label="initial DOE")
        if bo.any():
            it = order[bo] - n0 + 1
            sc = ax.scatter(X[bo, 0], Y[bo], c=it, cmap="viridis", s=42,
                            edgecolor="k", linewidth=0.4, zorder=3, label="BO samples")
        if int(round(Xme[-1, 1])) == lv:
            ax.axvline(Xme[-1, 0], color="r", ls=":", lw=1.6, label="recommended x*")
        if lv == glv:
            ax.plot(gx1, gmin, "m*", ms=15, zorder=4, label="global optimum")
        ax.set_title(f"level {lv}  (value {VAR_FCTR[lv-1]:g}, noise×{NOISE_MULS[lv-1]:g})",
                     fontsize=9)
        ax.set_xlabel("$x_1$"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("objective $y$")
    h, l = axes[0].get_legend_handles_labels()
    if bo.any():
        fig.colorbar(sc, ax=axes, label="BO iteration", shrink=0.7, pad=0.01)
    fig.legend(h, l, loc="upper center", ncol=6, fontsize=8, bbox_to_anchor=(0.5, 1.06))
    plabel = "" if param != param else f"={param:g}"
    fig.suptitle(f"Single BO run — {acf}{plabel}, n_rep={nrep}, seed={seed}", y=1.10)

    outdir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.path.dirname(__file__), "plots", "single_runs")
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir,
                       f"single_run_{acf}{'' if param!=param else '_'+str(param)}_nr{nrep}_s{seed}.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
