"""Compare LHS-flavored initial-DOE x1 schemes (2 points per category) that keep
the stratified/space-filling LHS character while pulling points off the bounds.
All faithfully mimic MATLAB lhsdesign(...,'criterion','none') = pure random LHS."""
import numpy as np
import matplotlib.pyplot as plt

LB, UB = -5.0, 10.0
R = UB - LB
N_LV, N_TR_LV = 5, 2
N_SEEDS = 30
GT_X1, GT_LV = 3.18, 2


def lhs_unit(n, rng):
    """Pure LHS on [0,1]: one uniform-random point per equal bin (no maximin)."""
    return (rng.permutation(n) + rng.random(n)) / n


# --- schemes -------------------------------------------------------------
def s_current(rng):                      # A: maximin over full range (pins to bounds)
    best, bestd = None, -1.0
    for _ in range(200):
        p = (rng.permutation(N_TR_LV) + rng.random(N_TR_LV)) / N_TR_LV
        d = np.min(np.abs(p[:, None] - p[None, :]) + np.eye(N_TR_LV) * 9)
        if d > bestd:
            bestd, best = d, p
    return np.sort(best) * R + LB


def s_lhs(rng):                          # 1: plain LHS, full range
    return np.sort(lhs_unit(N_TR_LV, rng)) * R + LB


def s_lhs_inset(rng, buf=0.15):          # 2: LHS on inset range (edge buffer)
    lo, hi = LB + buf * R, UB - buf * R
    return np.sort(lhs_unit(N_TR_LV, rng)) * (hi - lo) + lo


def s_lhs_drop(rng):                     # 3: (n+2)-bin LHS, drop the 2 boundary bins
    u = np.sort(lhs_unit(N_TR_LV + 2, rng))        # 4 bins, one pt each
    return u[1:-1] * R + LB                         # keep the interior 2


SCHEMES = [
    ("A — current: maximin, full range", s_current, "pinned to bounds (~-5, ~10)"),
    ("1 — plain LHS, full range", s_lhs, "stratified halves; can still graze bounds"),
    ("2 — LHS + edge buffer (inset 15%)", s_lhs_inset, "LHS within [-2.75, 7.75]"),
    ("3 — 4-bin LHS, drop boundary bins", s_lhs_drop, "keep interior 2 of 4 LHS pts"),
]


def collect(fn):
    rows = []
    for seed in range(1, N_SEEDS + 1):
        rng = np.random.default_rng(seed)
        for lv in range(1, N_LV + 1):
            for x1 in fn(rng):
                rows.append((lv, x1))
    return np.array(rows)


def main():
    thirds = [LB + R / 3, LB + 2 * R / 3]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
    jrng = np.random.default_rng(0)
    for ax, (name, fn, sub) in zip(axes.ravel(), SCHEMES):
        pts = collect(fn)
        jit = (jrng.random(len(pts)) - 0.5) * 0.5
        ax.scatter(pts[:, 1], pts[:, 0] + jit, s=13, alpha=0.5,
                   color="#4a90d9", edgecolor="none")
        for b in (LB, UB):
            ax.axvline(b, color="k", lw=1, alpha=0.4)
        for t in thirds:
            ax.axvline(t, color="green", ls="--", lw=1.1, alpha=0.6)
        ax.plot(GT_X1, GT_LV, "r*", ms=16, zorder=5)
        ax.set_title(f"{name}\n{sub}", fontsize=10)
        ax.set_yticks(range(1, N_LV + 1)); ax.set_ylim(0.3, N_LV + 0.7)
        ax.grid(axis="x", alpha=0.25)
    for ax in axes[1]:
        ax.set_xlabel("$x_1$")
    for ax in axes[:, 0]:
        ax.set_ylabel("categorical level")
    fig.suptitle("LHS-flavored initial-DOE schemes — 2 points/category, 30 seeds "
                 "(green dashed = 1/3 & 2/3 marks, ★ = true optimum)", y=1.0, fontsize=12)
    fig.tight_layout()
    fig.savefig("doe_lhs_options.png", dpi=140, bbox_inches="tight")
    print("saved doe_lhs_options.png")


if __name__ == "__main__":
    main()
