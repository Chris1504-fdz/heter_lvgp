#!/usr/bin/env python
"""
Log the study to TensorBoard (file-based, append-only -> robust for this long,
restart-prone job; no online service / single-active-run constraints).

One-shot: reads results/<acq>/nrep<NN>/seed*.mat and writes to study/tb/:
  progress/      pct_complete, runs_completed, conditions_finished  (vs runs done)
  conditions/<tag>/  best_y_mean & true_regret_mean per iteration (one run each)
  images/        the generated plot PNGs

Run periodically (see tb_loop.sh). Only the `tensorboard` server is long-lived,
and it just serves the files -- restartable with zero data loss.
"""
import os, json, glob
import numpy as np
import scipy.io
import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

CONFIG_ORDER = ["lcb", "pi", "ei", "haei_g0.5", "haei_g1", "haei_g5",
                "anpei_b0.2", "anpei_b0.5", "anpei_b0.8",
                "rahbo_a0.5", "rahbo_a1", "rahbo_a5"]

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
TB = os.path.join(HERE, "tb")
MANIFEST = os.path.join(HERE, ".tb_conditions.json")
VAR_FCTR = np.array([15, 2, 8, 0, 10.])
TOTAL = 1080
KNOB = {"haei": "γ", "rahbo": "α", "anpei": "β"}


def f_true(x1, x2):
    return ((x2 - 5.1/(4*np.pi**2)*x1**2 + 5/np.pi*x1 - 6)**2
            + 10*(1 - 1/(8*np.pi))*np.cos(x1) + 10)


def gt():
    x1 = np.linspace(-5, 10, 4000)
    return min(f_true(x1, v).min() for v in VAR_FCTR)


def label(acf, param):
    return acf.upper() if param != param else f"{acf}({KNOB.get(acf,'p')}={param:g})"


def conditions():
    g = {}
    for p in glob.glob(os.path.join(RESULTS, "*", "nrep*", "seed*.mat")):
        g.setdefault(os.path.dirname(p), []).append(p)
    return g


def load(path):
    m = scipy.io.loadmat(path)
    if "Y_min_history" not in m:
        return None
    meta = m["meta"][0, 0]
    return dict(acf=str(meta["acf"][0]), param=float(np.ravel(meta["acf_param"])[0]),
                n_rep=int(np.ravel(meta["n_rep"])[0]),
                y=np.ravel(m["Y_min_history"]).astype(float),
                Xme=np.atleast_2d(m["X_min_est"]).astype(float) if "X_min_est" in m else None)


def log_conditions(GT):
    logged = set(json.load(open(MANIFEST))) if os.path.exists(MANIFEST) else set()
    new = 0
    for cond_dir, mats in conditions().items():
        tag = f"{os.path.basename(os.path.dirname(cond_dir))}__nrep{cond_dir[-2:]}"
        if tag in logged or len(mats) < 30:
            continue
        rs = [load(p) for p in sorted(mats)]; rs = [r for r in rs if r]
        if not rs:
            continue
        L = min(len(r["y"]) for r in rs)
        by = np.array([r["y"][:L] for r in rs]).mean(0)
        reg = []
        for r in rs:
            if r["Xme"] is None:
                continue
            cur = [abs(float(f_true(np.array([xr[0]]), VAR_FCTR[int(round(xr[1]))-1])[0]) - GT)
                   for xr in r["Xme"][:L]]
            reg.append(cur)
        reg = np.array(reg).mean(0) if reg else None
        name = label(rs[0]["acf"], rs[0]["param"])
        w = SummaryWriter(log_dir=os.path.join(TB, "conditions", f"{name}_nrep{rs[0]['n_rep']}"))
        for i in range(L):
            w.add_scalar("best_y_mean", by[i], i + 1)
            if reg is not None:
                w.add_scalar("true_regret_mean", reg[i], i + 1)
        w.close()
        logged.add(tag); new += 1
        print(f"tb: logged condition {tag}")
    json.dump(sorted(logged), open(MANIFEST, "w"))
    return new


def log_progress():
    done = sum(len(v) for v in conditions().values())
    cond_done = sum(1 for v in conditions().values() if len(v) >= 30)
    w = SummaryWriter(log_dir=os.path.join(TB, "progress"))
    w.add_scalar("pct_complete", 100.0*done/TOTAL, done)
    w.add_scalar("runs_completed", done, done)
    w.add_scalar("conditions_finished", cond_done, done)
    w.close()
    return done, cond_done


def log_images():
    w = SummaryWriter(log_dir=os.path.join(TB, "images"))
    for png in sorted(glob.glob(os.path.join(HERE, "plots", "**", "*.png"), recursive=True)):
        tag = os.path.relpath(png, os.path.join(HERE, "plots"))[:-4]
        img = mpimg.imread(png)[..., :3]            # drop alpha
        w.add_image(tag, img, 0, dataformats="HWC")
    w.close()


def progress_figure():
    """Overall done/1080 bar + per-acquisition completion breakdown."""
    per = {c: 0 for c in CONFIG_ORDER}
    for cond_dir, mats in conditions().items():
        acf = os.path.basename(os.path.dirname(cond_dir))
        per[acf] = per.get(acf, 0) + len(mats)
    total = sum(per.values())

    fig = plt.figure(figsize=(9, 6.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 4], hspace=0.35)

    # --- overall progress bar ---
    ax0 = fig.add_subplot(gs[0])
    ax0.barh([0], [TOTAL], color="0.9", edgecolor="0.6")
    ax0.barh([0], [total], color="#2e7d32")
    ax0.set_xlim(0, TOTAL); ax0.set_ylim(-0.5, 0.5); ax0.set_yticks([])
    ax0.set_title(f"Experiments completed:  {total} / {TOTAL}   ({100*total/TOTAL:.1f}%)",
                  fontsize=13, fontweight="bold")
    ax0.text(total, 0, f" {total}", va="center", ha="left", fontsize=10)

    # --- per-acquisition breakdown (each /90) ---
    ax1 = fig.add_subplot(gs[1])
    names = CONFIG_ORDER[::-1]
    vals = [per[c] for c in names]
    colors = ["#2e7d32" if v >= 90 else ("#f9a825" if v > 0 else "#c62828") for v in vals]
    y = range(len(names))
    ax1.barh(list(y), [90]*len(names), color="0.93", edgecolor="0.7")
    ax1.barh(list(y), vals, color=colors)
    for i, (c, v) in enumerate(zip(names, vals)):
        ax1.text(91, i, f"{v}/90", va="center", ha="left", fontsize=8)
    ax1.set_yticks(list(y)); ax1.set_yticklabels(names, fontsize=8)
    ax1.set_xlim(0, 100); ax1.set_xlabel("runs completed (of 90 = 3 n_rep × 30 seeds)")
    ax1.set_title("By acquisition  (green=done, amber=partial, red=none)", fontsize=10)
    fig.suptitle("Study progress", y=0.98, fontsize=11)
    return fig, total


def main():
    os.makedirs(TB, exist_ok=True)
    GT = float(gt())
    n = log_conditions(GT)
    done, cond_done = log_progress()
    # completion figure: save to disk (fresh each cycle) AND log live to TB
    fig, total = progress_figure()
    fig.savefig(os.path.join(HERE, "plots", "main", "0_progress.png"),
                dpi=140, bbox_inches="tight")
    w = SummaryWriter(log_dir=os.path.join(TB, "progress"))
    w.add_figure("completion", fig, global_step=total)
    w.close(); plt.close(fig)
    log_images()   # picks up the fresh 0_progress.png too
    print(f"tb: {done}/{TOTAL} runs, {cond_done} conditions done ({n} newly logged)")


if __name__ == "__main__":
    main()
