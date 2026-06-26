#!/usr/bin/env python
"""
Live TensorBoard monitor for the sweep PROGRESS (not the model results).

Periodically scans results/<acq_tag>/nrep<NN>/seed<NN>.mat and logs, with the
x-axis = elapsed minutes:
  progress/completed, progress/percent, progress/remaining
  time/elapsed_min, time/eta_min, time/finish_in_min
  rate/runs_per_min (overall), rate/runs_per_min_recent (last window)
  runtime/avg_sec_per_run, runtime/last_run_sec
  workers/active_matlab
  per_acq/<tag>            completed count out of target (0..90), 12 series
  per_acq_pct/<tag>        percent complete per acquisition

Robust/restartable: file-based, append-only. Exits when all runs are done.

  python tb_monitor.py --logdir tb/run1 --interval 30
"""
import os, glob, time, argparse, collections
import scipy.io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
CONFIG_ORDER = ["lcb", "pi", "ei", "haei_g0.5", "haei_g1", "haei_g5",
                "anpei_b0.2", "anpei_b0.5", "anpei_b0.8",
                "rahbo_a0.5", "rahbo_a1", "rahbo_a5"]
LABELS = ["LCB", "PI", "EI", "HAEI γ0.5", "HAEI γ1", "HAEI γ5",
          "ANPEI β0.2", "ANPEI β0.5", "ANPEI β0.8",
          "RAHBO α0.5", "RAHBO α1", "RAHBO α5"]


def build_bar_figure(per_acq, target, done, total, elapsed_min, eta_min):
    counts = [per_acq[t] for t in CONFIG_ORDER]
    fig, ax = plt.subplots(figsize=(12, 5.5))
    colors = ["#2e7d32" if c >= target else "#4a90d9" for c in counts]
    bars = ax.bar(range(len(counts)), counts, color=colors, edgecolor="k", lw=0.4)
    ax.axhline(target, color="crimson", ls="--", lw=1.8, label=f"target = {target}")
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width()/2, c + target*0.015, str(int(c)),
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS, rotation=40, ha="right", fontsize=9)
    ax.set_ylim(0, target * 1.12); ax.set_ylabel("completed runs")
    pct = 100 * done / total if total else 0
    eta_txt = f"{eta_min:.0f} min" if eta_min == eta_min else "—"
    ax.set_title(f"Sweep progress — {done}/{total} ({pct:.0f}%)   "
                 f"·  elapsed {elapsed_min:.0f} min  ·  ETA {eta_txt}   "
                 f"·  {sum(c >= target for c in counts)}/{len(counts)} configs done")
    ax.legend(loc="lower right"); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def count_active_matlab():
    try:
        import subprocess
        out = subprocess.run(["pgrep", "-f", "run_chunk"],
                             capture_output=True, text=True).stdout
        return len([l for l in out.splitlines() if l.strip()])
    except Exception:
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", default=os.path.join(HERE, "tb", "sweep"))
    ap.add_argument("--total", type=int, default=1080)
    ap.add_argument("--target-per-acq", type=int, default=90)
    ap.add_argument("--interval", type=float, default=30.0)
    args = ap.parse_args()

    w = SummaryWriter(args.logdir)
    print(f"tb_monitor -> {args.logdir} | total={args.total} | every {args.interval:.0f}s")
    t0 = time.time()
    seen = set()
    runtime_sum, runtime_n, last_run_sec = 0.0, 0, 0.0
    recent = collections.deque()           # (t, completed) for windowed rate
    window_s = 300.0

    while True:
        now = time.time()
        elapsed = now - t0
        files = glob.glob(os.path.join(RESULTS, "**", "*.mat"), recursive=True)
        done = len(files)

        # per-acq counts
        per_acq = {tag: 0 for tag in CONFIG_ORDER}
        for f in files:
            tag = os.path.relpath(f, RESULTS).split(os.sep)[0]
            if tag in per_acq:
                per_acq[tag] += 1

        # accumulate per-run runtime from newly-seen files only
        for f in files:
            if f in seen:
                continue
            seen.add(f)
            try:
                rt = float(scipy.io.loadmat(f)["meta"][0, 0]["runtime"].ravel()[0])
                runtime_sum += rt; runtime_n += 1; last_run_sec = rt
            except Exception:
                pass

        # rates / ETA  (guard the warm-up: need a real time window before trusting)
        recent.append((now, done))
        while recent and now - recent[0][0] > window_s:
            recent.popleft()
        span = recent[-1][0] - recent[0][0] if len(recent) >= 2 else 0.0
        rate = done / (elapsed / 60.0) if elapsed > 30 else float("nan")    # overall runs/min
        if span >= 45 and (recent[-1][1] - recent[0][1]) > 0:
            rate_recent = (recent[-1][1] - recent[0][1]) / (span / 60.0)    # windowed runs/min
        elif elapsed >= 90 and done > 0:
            rate_recent = done / (elapsed / 60.0)
        else:
            rate_recent = float("nan")                                     # too early to estimate
        remaining = max(args.total - done, 0)
        eta_min = remaining / rate_recent if rate_recent and rate_recent > 0 else float("nan")

        step = int(elapsed)                  # x-axis ~ elapsed seconds
        wt = now
        # THE single plot: 12 bars + target line at 90, updated each scan.
        fig = build_bar_figure(per_acq, args.target_per_acq, done, args.total,
                               elapsed / 60.0, eta_min)
        w.add_figure("sweep_progress", fig, global_step=step)
        plt.close(fig)
        # a few handy summary scalars (kept minimal, not the messy per-acq spam)
        w.add_scalar("summary/percent_complete", 100.0 * done / args.total, step, wt)
        w.add_scalar("summary/elapsed_min", elapsed / 60.0, step, wt)
        w.add_scalar("summary/eta_min", eta_min, step, wt)
        w.flush()

        print(f"[{elapsed/60:6.1f} min] {done:4d}/{args.total} "
              f"({100*done/args.total:4.1f}%)  rate={rate_recent:4.1f}/min  "
              f"eta={eta_min:5.1f} min  workers={count_active_matlab()}")

        if done >= args.total:
            w.add_text("status", f"DONE — {done}/{args.total} in {elapsed/60:.1f} min", step)
            w.flush(); w.close()
            print("all runs complete; monitor exiting.")
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
