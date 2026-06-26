#!/usr/bin/env python
"""
Parameter study for the PER-CATEGORY GP Bayesian optimization (study_v2_gp, "Part 2").

Same grid as the MATLAB LVGP study (study_v2): 12 acquisition configs x n_rep{3,5,10} x
N seeds. Each cell runs ONE BO with independent GPs per category (utils/bo.run_bo) and
saves results/<acf_tag>/nrep<NN>/seed<NN>.npz, name-for-name with study_driver.m.

Pure Python -- NO MATLAB license, NO Xvfb, NO prefdir juggling. Every cell is an
independent process in a ProcessPoolExecutor; each worker pins torch to 1 thread
(~ MATLAB -singleCompThread) and outer parallelism is handled here.

  python run_sweep.py --seeds 30 --num-iter 30 --workers 18     # full 1080-run grid
  python run_sweep.py --toy --workers 8                         # quick 5-config probe
  python run_sweep.py --only rahbo haei --seeds 5               # restrict acquisitions
  python run_sweep.py --collect-only                            # just rebuild the CSV

Resumable: cells whose .npz already exists are skipped.
"""
import os
# Pin BLAS / OpenMP to 1 thread per process BEFORE numpy/torch import, so the N outer
# workers don't each spawn a linear-algebra thread pool and oversubscribe the cores
# (outer parallelism is the ProcessPoolExecutor below; cf. MATLAB -singleCompThread).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse, sys, itertools, glob
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                                      # so `import utils` works from anywhere
RESULTS = os.path.join(HERE, "results")

from utils import problem                                     # light: numpy/scipy only

# The 12 chosen acquisition configs (acf, knob_value); knob is NaN when unused.
ACQ_CONFIGS = list(problem.CONFIG_ORDER)
N_REP_LIST = [3, 5, 10]

# Small representative subset for a quick probe (one per acquisition family).
TOY_CONFIGS = [
    ("ei", float("nan")),
    ("lcb", float("nan")),
    ("haei", 1.0),
    ("anpei", 0.5),
    ("rahbo", 1.0),
]


def cell_paths(acf, param, n_rep, seed):
    """Nested layout results/<acf_tag>/nrep<NN>/seed<NN>.npz + a print tag.

    Saved as .npz with the SAME field names as study_v2/study_driver.m, so the reused
    StudyResults gallery (whose loader reads both .npz and .mat) plots these directly and
    the LVGP study (.mat) overlays for the head-to-head comparison."""
    d = os.path.join(RESULTS, problem.acf_tag(acf, param), f"nrep{n_rep:02d}")
    base = f"seed{seed:02d}"
    tag = f"{problem.acf_tag(acf, param)}/nrep{n_rep:02d}/{base}"
    return d, os.path.join(d, base + ".npz"), tag


def run_one(args):
    """Worker: run one BO cell and save its .npz. Returns (tag, status)."""
    acf, param, n_rep, seed, num_iter = args
    d, out, tag = cell_paths(acf, param, n_rep, seed)
    os.makedirs(d, exist_ok=True)
    if os.path.exists(out):
        return tag, "skip(exists)"
    import warnings; warnings.filterwarnings("ignore")        # silence gpytorch noise-clamp spam
    import torch
    torch.set_num_threads(1)                                  # ~ MATLAB -singleCompThread
    from utils.bo import run_bo
    try:
        res = run_bo(acf, param, n_rep, seed, num_iter)
    except Exception as e:                                    # don't kill the pool on one bad cell
        import traceback
        with open(out + ".err", "w") as fh:
            fh.write(traceback.format_exc())
        return tag, f"FAIL({type(e).__name__}: {e})"
    meta = res.pop("meta")
    np.savez(out, meta=np.array(meta, dtype=object), **res)  # study_driver.m field schema
    return tag, f"ok(final_y={res['Y_min_history'][-1]:.4g}, {meta['runtime']:.1f}s)"


def collect():
    """Load every run (.npz) into a tidy CSV: best_y / y_min_est / true_regret per iter."""
    import pandas as pd
    gtmin = problem.ground_truth_min()
    rows = []
    for f in glob.glob(os.path.join(RESULTS, "**", "*.npz"), recursive=True):
        z = np.load(f, allow_pickle=True)
        if "Y_min_history" not in z:
            continue
        meta = z["meta"].item()
        y = np.ravel(z["Y_min_history"]).astype(float)
        ymin_est = np.ravel(z["Y_min_est"]).astype(float)
        Xme = np.atleast_2d(z["X_min_est"]).astype(float)    # (num_iter, [x1, level])
        # true-regret at the recommended optimum each iter (noise-free f_true minus global min)
        treg = np.array([problem.f_true_level(Xme[i, 0], int(round(Xme[i, 1]))) - gtmin
                         for i in range(Xme.shape[0])])
        for it in range(len(y)):
            rows.append(dict(acf=str(meta["acf"]), param=float(meta["acf_param"]),
                             n_rep=int(meta["n_rep"]), seed=int(meta["seed"]),
                             iter=it + 1, best_y=y[it],
                             y_min_est=ymin_est[it] if it < len(ymin_est) else np.nan,
                             true_regret=treg[it] if it < len(treg) else np.nan,
                             runtime=float(meta["runtime"])))
    df = pd.DataFrame(rows).sort_values(["acf", "param", "n_rep", "seed", "iter"])
    csv = os.path.join(HERE, "sweep_results.csv")
    df.to_csv(csv, index=False)
    return df, csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30, help="number of seeds per condition")
    ap.add_argument("--num-iter", type=int, default=30, help="BO iterations per run")
    ap.add_argument("--workers", type=int, default=18, help="concurrent processes")
    ap.add_argument("--collect-only", action="store_true", help="just rebuild the CSV")
    ap.add_argument("--toy", action="store_true", help="quick probe: 5-config subset")
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these acquisition names")
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    if args.collect_only:
        df, csv = collect(); print(f"Collected {len(df)} rows -> {csv}"); return

    configs = TOY_CONFIGS if args.toy else ACQ_CONFIGS
    if args.only:
        configs = [(a, p) for (a, p) in configs if a in args.only]
    grid = [(acf, param, n_rep, seed, args.num_iter)
            for (acf, param), n_rep, seed
            in itertools.product(configs, N_REP_LIST, range(1, args.seeds + 1))]
    print(f"{len(grid)} runs | {args.workers} workers | num_iter={args.num_iter}", flush=True)

    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, g): g for g in grid}
        for fut in as_completed(futs):
            tag, status = fut.result(); done += 1
            print(f"[{done}/{len(grid)}] {tag}: {status}", flush=True)

    df, csv = collect()
    print(f"\nCollected {len(df)} rows -> {csv}")


if __name__ == "__main__":
    main()
