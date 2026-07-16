"""
validate_torch.py -- distributional equivalence check: torch LVGP vs MATLAB LVGP on branin_hetero.

Pairs lvgp_torch  <-> standard_LVGP   (noise-blind: ei/lcb/pi)
      heter_lvgp_torch <-> heter_LVGP  (all 6 acquisitions)

Why DISTRIBUTIONAL, not cell-by-cell: the shared SLHD initial design is byte-identical across engines
(CRN via doe_cache), but the BO *iteration* noise is not -- python draws default_rng([seed,1]),
MATLAB draws rng(seed). So even a perfect reimplementation diverges after iteration 1. Equivalence is
therefore judged on the 30-seed DISTRIBUTION of the headline metric, per (acquisition, n_rep).

Headline metric: best-found noiseless objective = min over sampled points of f_true_sampled
(the user's "true_best_sampled"). Lower is better. Regret = metric - f*, but f* is a shared constant,
so comparing the metric distributions IS comparing regret.

Tests per (acq, n_rep):
  - Wilcoxon signed-rank (paired by seed -- leverages the shared initial design; more powerful)
  - Mann-Whitney U (unpaired -- robust if pairing beyond the initial design is doubted)
Verdict EQUIVALENT if neither test rejects at alpha=0.05 (two-sided) AND medians are within a small
relative tolerance. A significant result in EITHER direction fails -- a drop-in must not be better OR
worse than what it replaces.
"""
import os, sys, glob, argparse
import numpy as np
import scipy.io as sio
from scipy.stats import wilcoxon, mannwhitneyu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import problems as P

PAIRS = [("lvgp_torch", "standard_LVGP", ["ei", "lcb", "pi"]),
         ("heter_lvgp_torch", "heter_LVGP", ["ei", "lcb", "pi", "haei", "anpei", "rahbo"])]
ALPHA = 0.05
FUNCTION = "branin_hetero"


def _load(path):
    if path.endswith(".mat"):
        return sio.loadmat(path, simplify_cells=True)
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def _acq_dir(root, model, acq):
    """The stored acq subdir may carry a param suffix (haei_g0.5, anpei_b0.8, rahbo_a0.5)."""
    base = os.path.join(root, FUNCTION, model)
    if not os.path.isdir(base):
        return None
    for d in os.listdir(base):
        if d == acq or d.startswith(acq + "_"):
            return os.path.join(base, d)
    return None


def _metric_and_traj(cell):
    """best-found noiseless value (scalar) and its running-min trajectory over sampled points."""
    ft = np.asarray(cell["f_true_sampled"], float).ravel()
    return float(np.min(ft)), np.minimum.accumulate(ft)


def _collect(root, model, acq, nrep):
    d = _acq_dir(root, model, acq)
    if d is None:
        return {}
    out = {}
    for f in glob.glob(os.path.join(d, f"nrep{nrep:02d}", "seed*")):
        try:
            c = _load(f)
            seed = int(np.ravel(c["meta"].item()["seed"] if hasattr(c["meta"], "item") else c["meta"]["seed"]))
            out[seed] = _metric_and_traj(c)
        except Exception as e:
            print(f"  skip {f}: {e}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--torch-root", default="results")
    ap.add_argument("--matlab-root", default="results")
    ap.add_argument("--nreps", nargs="*", type=int, default=[3, 10])
    ap.add_argument("--fig", default="plots/validate_torch_branin.png")
    ap.add_argument("--csv", default="plots/validate_torch_branin.csv")
    args = ap.parse_args()

    rows = []
    panels = []
    for tmodel, mmodel, acqs in PAIRS:
        for acq in acqs:
            for nrep in args.nreps:
                T = _collect(args.torch_root, tmodel, acq, nrep)
                M = _collect(args.matlab_root, mmodel, acq, nrep)
                seeds = sorted(set(T) & set(M))
                if len(seeds) < 5:
                    rows.append((tmodel, acq, nrep, len(seeds), np.nan, np.nan,
                                 np.nan, np.nan, np.nan, "INSUFFICIENT"))
                    continue
                tv = np.array([T[s][0] for s in seeds])
                mv = np.array([M[s][0] for s in seeds])
                tmed, mmed = np.median(tv), np.median(mv)
                tiqr = np.subtract(*np.percentile(tv, [75, 25]))
                miqr = np.subtract(*np.percentile(mv, [75, 25]))
                # paired (shared initial design) + unpaired
                try:
                    p_w = wilcoxon(tv, mv, zero_method="zsplit").pvalue if np.any(tv != mv) else 1.0
                except ValueError:
                    p_w = 1.0
                p_mw = mannwhitneyu(tv, mv, alternative="two-sided").pvalue
                scale = max(abs(mmed), 1e-9)
                rel = abs(tmed - mmed) / scale
                equiv = (p_w > ALPHA) and (p_mw > ALPHA) and (rel < 0.15)
                verdict = "EQUIVALENT" if equiv else ("DIFFER(p_w=%.3f,p_mw=%.3f)" % (p_w, p_mw))
                rows.append((tmodel, acq, nrep, len(seeds), tmed, mmed,
                             tiqr, miqr, p_w, p_mw, rel, verdict))
                panels.append((f"{acq} nrep{nrep}", seeds, T, M))

    # ---- table ----
    hdr = ("model", "acq", "nrep", "n", "torch_med", "matlab_med",
           "torch_iqr", "matlab_iqr", "p_wilcox", "p_mannwhit", "rel_gap", "verdict")
    print("\n" + " ".join(f"{h:>11s}" for h in hdr))
    with open(args.csv, "w") as fh:
        fh.write(",".join(hdr) + "\n")
        for r in rows:
            cells = [str(r[0]), str(r[1]), str(r[2]), str(r[3])] + \
                    ["%.4g" % x if isinstance(x, float) else str(x) for x in r[4:]]
            fh.write(",".join(cells) + "\n")
            disp = [f"{r[0][:11]:>11s}", f"{r[1]:>11s}", f"{r[2]:>11d}", f"{r[3]:>11d}"]
            disp += [f"{x:>11.4g}" if isinstance(x, float) and np.isfinite(x) else f"{'--':>11s}"
                     for x in r[4:-1]]
            disp += [f"{r[-1]:>11s}"]
            print(" ".join(disp))

    n_eq = sum(1 for r in rows if r[-1] == "EQUIVALENT")
    n_tot = sum(1 for r in rows if r[-1] not in ("INSUFFICIENT",))
    print(f"\nVERDICT: {n_eq}/{n_tot} (acq,nrep) cells EQUIVALENT at alpha={ALPHA}")

    # ---- trajectory figure ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        k = len(panels)
        if k:
            ncol = 4; nrow = int(np.ceil(k / ncol))
            fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow), squeeze=False)
            for ax, (title, seeds, T, M) in zip(axes.ravel(), panels):
                for src, col in [(T, "C0"), (M, "C1")]:
                    trajs = np.array([src[s][1] for s in seeds])
                    L = min(t.shape[0] for t in trajs)
                    trajs = np.array([t[:L] for t in trajs])
                    med = np.median(trajs, 0)
                    lo, hi = np.percentile(trajs, [25, 75], 0)
                    x = np.arange(L)
                    ax.plot(x, med, col, lw=1.5, label=("torch" if col == "C0" else "matlab"))
                    ax.fill_between(x, lo, hi, color=col, alpha=0.15)
                ax.set_title(title, fontsize=9); ax.legend(fontsize=7)
            for ax in axes.ravel()[k:]:
                ax.axis("off")
            fig.suptitle("branin_hetero: best-found noiseless f (median +/- IQR, 30 seeds)", y=1.0)
            fig.tight_layout()
            os.makedirs(os.path.dirname(args.fig), exist_ok=True)
            fig.savefig(args.fig, dpi=120, bbox_inches="tight")
            print(f"figure -> {args.fig}")
    except Exception as e:
        print(f"(figure skipped: {e})")


if __name__ == "__main__":
    main()
