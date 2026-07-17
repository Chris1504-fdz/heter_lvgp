"""
compare_lvgp.py -- native Python LVGP vs MATLAB LVGP on branin_hetero, across four axes:
  1. RESULTS    : best-found noiseless value distributions (median/IQR, rel gap, Mann-Whitney).
  2. CONVERGENCE: running-min-of-f_true trajectories, median +/- IQR bands per acquisition.
  3. COVERAGE   : how the two engines sample -- per-level sampling fractions + x-spread in the
                  optimal level (level 2). Same shared initial design (CRN); iteration noise differs.
  4. LATENT     : the fitted latent embedding z (5 levels x 2). MATLAB stores it per cell; native is
                  refit on the matched cell's final training data. Both use the SAME identifiability
                  gauge (level 1 at origin, level-2 2nd coord = 0), so z is directly comparable up to
                  a reflection of the 2nd axis (aligned before scoring).
"""
import os, sys, glob, argparse
import numpy as np
import scipy.io as sio
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import problems as P
from utils.models.hetero_lvgp_native import HeterLVGPNative, _z_matrix

NLV = 5
OPT_LEVEL = 2


def _load(p):
    if p.endswith(".mat"):
        return sio.loadmat(p, simplify_cells=True)
    with np.load(p, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def _meta(c):
    m = c["meta"]
    return m.item() if hasattr(m, "item") and m.shape == () else m


def _acqdir(root, model, acq):
    base = os.path.join(root, "branin_hetero", model)
    if not os.path.isdir(base):
        return None
    for d in os.listdir(base):
        if d == acq or d.startswith(acq + "_"):
            return os.path.join(base, d)
    return None


def _cells(root, model, acq, nrep):
    d = _acqdir(root, model, acq)
    out = {}
    if not d:
        return out
    for f in glob.glob(os.path.join(d, f"nrep{nrep:02d}", "seed*")):
        if not f.endswith((".npz", ".mat")):
            continue
        c = _load(f)
        out[int(_meta(c)["seed"])] = c
    return out


def _best(c):
    return float(np.min(np.asarray(c["f_true_sampled"], float)))


def _traj(c):
    return np.minimum.accumulate(np.asarray(c["f_true_sampled"], float))


def _levels_sampled(c):
    X = np.asarray(c["X_sampled"], float)
    return X[:, -1].astype(int)


def _native_latent(c):
    """Refit HeterLVGPNative on the cell's final training data, return z (NLV x 2)."""
    X = np.asarray(c["X_sampled"], float)
    y = np.asarray(c["Y_sampled"], float).ravel()
    v = np.asarray(c["Y_var_sampled"], float).ravel()
    data = {}
    for lv in range(1, NLV + 1):
        m = X[:, -1].astype(int) == lv
        if m.sum() == 0:
            continue
        data[lv] = {"X": X[m, :-1], "y_mean": y[m], "y_var": v[m]}
    model = HeterLVGPNative.fit(data, needs_r=False, bounds=P.get("branin_hetero").bounds)
    return _z_matrix(model._f["zf"], model._f["n_lvs"]), sorted(data)


def _align_reflection(zn, zm):
    """Both share the canonical gauge; remaining freedom is a sign flip of the 2nd axis. Pick it."""
    best, bz = None, zn
    for s in (1.0, -1.0):
        z = zn.copy(); z[:, 1] *= s
        err = float(np.sqrt(((z - zm) ** 2).sum(1)).mean())
        if best is None or err < best:
            best, bz = err, z
    return bz, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--native-root", default="results")
    ap.add_argument("--matlab-root", default="results")
    ap.add_argument("--nrep", type=int, default=10)
    ap.add_argument("--acqs", nargs="*", default=["ei", "lcb", "pi", "haei", "anpei", "rahbo"])
    ap.add_argument("--latent-seeds", type=int, default=10)
    ap.add_argument("--fig", default="plots/compare_lvgp_branin.png")
    args = ap.parse_args()

    from scipy.stats import mannwhitneyu
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

    panels_traj = []
    print("\n" + "=" * 78)
    print("1. RESULTS  (best-found noiseless f; lower=better)   nrep=%d" % args.nrep)
    print("=" * 78)
    print(f"{'acq':7s}{'n':>4s}{'native med':>12s}{'matlab med':>12s}"
          f"{'nat IQR':>10s}{'mat IQR':>10s}{'rel gap':>9s}{'p(MW)':>9s}")
    cov_rows = []
    for acq in args.acqs:
        N = _cells(args.native_root, "heter_lvgp_native", acq, args.nrep)
        M = _cells(args.matlab_root, "heter_LVGP", acq, args.nrep)
        seeds = sorted(set(N) & set(M))
        if len(seeds) < 5:
            print(f"{acq:7s}{len(seeds):>4d}   (insufficient overlap)")
            continue
        nb = np.array([_best(N[s]) for s in seeds]); mb = np.array([_best(M[s]) for s in seeds])
        nmed, mmed = np.median(nb), np.median(mb)
        niqr = np.subtract(*np.percentile(nb, [75, 25])); miqr = np.subtract(*np.percentile(mb, [75, 25]))
        rel = abs(nmed - mmed) / max(abs(mmed), 1e-9)
        p = mannwhitneyu(nb, mb, alternative="two-sided").pvalue
        print(f"{acq:7s}{len(seeds):>4d}{nmed:>12.4f}{mmed:>12.4f}{niqr:>10.4f}{miqr:>10.4f}"
              f"{rel:>9.3f}{p:>9.3f}")
        # convergence panel data
        panels_traj.append((acq, seeds, N, M))
        # coverage: per-level sampling fraction + x-spread in optimal level
        for eng, cells in [("native", N), ("matlab", M)]:
            fr = np.zeros(NLV + 1)
            xspread = []
            for s in seeds:
                lv = _levels_sampled(cells[s])
                for L in range(1, NLV + 1):
                    fr[L] += np.mean(lv == L)
                Xo = np.asarray(cells[s]["X_sampled"], float)
                xo = Xo[Xo[:, -1].astype(int) == OPT_LEVEL, 0]
                if len(xo) > 1:
                    xspread.append(xo.std())
            fr = fr[1:] / len(seeds)
            cov_rows.append((acq, eng, fr, np.mean(xspread) if xspread else np.nan))

    # ---- 3. COVERAGE ----
    print("\n" + "=" * 78)
    print("3. COVERAGE  (mean fraction of samples at each level; opt=level %d)" % OPT_LEVEL)
    print("=" * 78)
    print(f"{'acq':7s}{'engine':8s}" + "".join(f"{'lv'+str(L):>8s}" for L in range(1, NLV + 1))
          + f"{'x-std@opt':>11s}")
    for acq, eng, fr, xs in cov_rows:
        print(f"{acq:7s}{eng:8s}" + "".join(f"{f:>8.2f}" for f in fr) + f"{xs:>11.3f}")

    # ---- 4. LATENT ----
    print("\n" + "=" * 78)
    print("4. LATENT EMBEDDING  (mean per-level |z_native - z_matlab| after reflection align)")
    print("=" * 78)
    lat_native_avg, lat_matlab_avg = {}, {}
    acq_lat = "lcb" if "lcb" in args.acqs else args.acqs[0]
    N = _cells(args.native_root, "heter_lvgp_native", acq_lat, args.nrep)
    M = _cells(args.matlab_root, "heter_LVGP", acq_lat, args.nrep)
    seeds = sorted(set(N) & set(M))[: args.latent_seeds]
    per_level_err, zn_all, zm_all = [], [], []
    for s in seeds:
        try:
            zn, lv_present = _native_latent(N[s])
            zm = np.asarray(_meta(M[s]) and M[s]["hyper"]["z"], float).reshape(NLV, 2)
        except Exception as e:
            continue
        zn_a, err = _align_reflection(zn, zm)
        per_level_err.append(np.sqrt(((zn_a - zm) ** 2).sum(1)))
        zn_all.append(zn_a); zm_all.append(zm)
    if per_level_err:
        pe = np.array(per_level_err).mean(0)
        print(f"  acq={acq_lat}, {len(per_level_err)} seeds")
        print("  " + "  ".join(f"lv{L+1}:{pe[L]:.3f}" for L in range(NLV))
              + f"   (mean {pe.mean():.3f})")
        zn_m = np.mean(zn_all, 0); zm_m = np.mean(zm_all, 0)
        print("  mean native z:\n" + "\n".join(f"    lv{L+1}: ({zn_m[L,0]:+.3f}, {zn_m[L,1]:+.3f})"
                                               for L in range(NLV)))
        print("  mean matlab z:\n" + "\n".join(f"    lv{L+1}: ({zm_m[L,0]:+.3f}, {zm_m[L,1]:+.3f})"
                                               for L in range(NLV)))

    # ---- figure: convergence bands + latent maps + coverage ----
    ncol = 3
    nrow = int(np.ceil((len(panels_traj) + 2) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.6 * nrow), squeeze=False)
    axf = axes.ravel()
    for ax, (acq, seeds, N, M) in zip(axf, panels_traj):
        for cells, col, lab in [(N, "C0", "native"), (M, "C1", "matlab")]:
            T = np.array([_traj(cells[s]) for s in seeds])
            L = min(t.shape[0] for t in T); T = np.array([t[:L] for t in T])
            x = np.arange(L)
            ax.plot(x, np.median(T, 0), col, lw=1.6, label=lab)
            lo, hi = np.percentile(T, [25, 75], 0)
            ax.fill_between(x, lo, hi, color=col, alpha=0.15)
        ax.set_title(f"convergence: {acq}", fontsize=9); ax.legend(fontsize=7)
        ax.set_xlabel("iteration"); ax.set_ylabel("best f_true")
    # latent map panel
    axl = axf[len(panels_traj)]
    if per_level_err:
        for L in range(NLV):
            axl.scatter(*zn_m[L], c="C0", marker="o", s=60)
            axl.scatter(*zm_m[L], c="C1", marker="x", s=70)
            axl.annotate(f"{L+1}", zn_m[L], fontsize=8)
        axl.scatter([], [], c="C0", marker="o", label="native")
        axl.scatter([], [], c="C1", marker="x", label="matlab")
        axl.set_title("latent embedding (mean)", fontsize=9); axl.legend(fontsize=7)
        axl.set_xlabel("z1"); axl.set_ylabel("z2")
    # coverage panel (level fractions, native vs matlab, averaged over acqs)
    axc = axf[len(panels_traj) + 1]
    frn = np.mean([fr for a, e, fr, x in cov_rows if e == "native"], 0)
    frm = np.mean([fr for a, e, fr, x in cov_rows if e == "matlab"], 0)
    w = 0.38; xL = np.arange(1, NLV + 1)
    axc.bar(xL - w/2, frn, w, color="C0", label="native")
    axc.bar(xL + w/2, frm, w, color="C1", label="matlab")
    axc.set_title("sampling fraction per level", fontsize=9); axc.legend(fontsize=7)
    axc.set_xlabel("level"); axc.set_xticks(xL)
    for ax in axf[len(panels_traj) + 2:]:
        ax.axis("off")
    fig.suptitle("Native Python LVGP vs MATLAB LVGP — branin_hetero (heteroscedastic)", y=1.0)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.fig), exist_ok=True)
    fig.savefig(args.fig, dpi=120, bbox_inches="tight")
    print(f"\nfigure -> {args.fig}")


if __name__ == "__main__":
    main()
