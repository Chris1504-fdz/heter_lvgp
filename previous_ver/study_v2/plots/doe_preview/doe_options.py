"""Compare two initial-DOE x1 placement schemes for the 2-point-per-category LHS.
Faithfully mimics MATLAB lhsdesign: Option A uses maximin (current), Option B
places points around the 1/3 and 2/3 marks (random LHS over the inner 2/3 range)."""
import numpy as np
import matplotlib.pyplot as plt

LB, UB = -5.0, 10.0
N_LV, N_TR_LV = 5, 2
N_SEEDS = 30
GT_X1, GT_LV = 3.18, 2            # true optimum location


def lhs_unit(n, rng):
    """One random point per equal bin (pure LHS, no optimization)."""
    perm = rng.permutation(n)
    return np.sort((perm + rng.random(n)) / n)


def lhs_maximin(n, rng, iters=1000):
    """Mimics lhsdesign(n,1,'iterations',iters): best min-distance LHS."""
    best, bestd = None, -1.0
    for _ in range(iters):
        pts = (rng.permutation(n) + rng.random(n)) / n
        d = np.min(np.abs(pts[:, None] - pts[None, :]) + np.eye(n) * 9)
        if d > bestd:
            bestd, best = d, pts
    return np.sort(best)


def option_A(rng):                       # current: maximin over full range
    u = lhs_maximin(N_TR_LV, rng)
    return u * (UB - LB) + LB


def option_B(rng):                       # thirds: points sit ON the 1/3 & 2/3 marks
    fracs = np.arange(1, N_TR_LV + 1) / (N_TR_LV + 1)   # 1/3, 2/3  -> 0.0, 5.0
    centers = LB + fracs * (UB - LB)
    gap = (UB - LB) / (N_TR_LV + 1)                     # 5.0
    jit = gap / 4.0                                     # +/-1.25 so they hug the marks
    return centers + (rng.random(N_TR_LV) - 0.5) * 2 * jit


def collect(opt):
    rows = []
    for seed in range(1, N_SEEDS + 1):
        rng = np.random.default_rng(seed)
        for lv in range(1, N_LV + 1):
            for x1 in opt(rng):
                rows.append((lv, x1))
    return np.array(rows)


def main():
    thirds = [LB + (UB - LB) / 3, LB + 2 * (UB - LB) / 3]   # 0.0, 5.0
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
    for ax, (name, opt, sub) in zip(
            axes,
            [("Option A — current (maximin, full range)", option_A,
              "2 points pushed to the bounds (~-5, ~10)"),
             ("Option B — thirds (points ON the 1/3 & 2/3 marks)", option_B,
              "2 points sit on the marks (~0, ~5), small jitter")]):
        pts = collect(opt)
        jit = (np.random.default_rng(0).random(len(pts)) - 0.5) * 0.5
        ax.scatter(pts[:, 1], pts[:, 0] + jit, s=14, alpha=0.5,
                   color="#4a90d9", edgecolor="none")
        for b in (LB, UB):
            ax.axvline(b, color="k", ls="-", lw=1, alpha=0.4)
        for t in thirds:
            ax.axvline(t, color="green", ls="--", lw=1.3, alpha=0.7)
        ax.plot(GT_X1, GT_LV, "r*", ms=20, zorder=5)
        ax.annotate("true opt", (GT_X1, GT_LV), textcoords="offset points",
                    xytext=(8, 8), color="r", fontsize=9)
        ax.set_title(f"{name}\n{sub}", fontsize=10)
        ax.set_xlabel("$x_1$"); ax.set_yticks(range(1, N_LV + 1))
        ax.set_ylim(0.3, N_LV + 0.7); ax.grid(axis="x", alpha=0.25)
    axes[0].set_ylabel("categorical level")
    fig.suptitle("Initial DOE x1 placement — 2 LHS points per category, "
                 f"{N_SEEDS} seeds overlaid (green dashed = 1/3 & 2/3 marks)",
                 y=1.02, fontsize=12)
    fig.tight_layout()
    out = "doe_options.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
