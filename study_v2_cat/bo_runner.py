#!/usr/bin/env python
"""
bo_runner.py -- run ONE per-category-GP BO cell and save its .npz.

Thin CLI around utils/bo.run_bo for single runs / debugging (the sweep calls run_bo
in-process instead). Saves to the same nested layout as run_sweep.py.

  python bo_runner.py --acf ei    --param nan --n-rep 10 --seed 1 --num-iter 30
  python bo_runner.py --acf haei  --param 1.0 --n-rep 5  --seed 3 --num-iter 30
  python bo_runner.py --acf rahbo --param 0.5 --n-rep 10 --seed 1 --out /tmp/probe.npz
"""
import argparse, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_sweep import cell_paths


def _parse_param(s):
    return float("nan") if s.lower() in ("nan", "none", "na") else float(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acf", required=True)
    ap.add_argument("--param", default="nan", help="family knob; 'nan' for ei/lcb/pi")
    ap.add_argument("--n-rep", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--num-iter", type=int, default=30)
    ap.add_argument("--out", default=None, help="override output .npz path")
    args = ap.parse_args()
    param = _parse_param(args.param)

    import torch
    torch.set_num_threads(1)
    from utils.bo import run_bo

    res = run_bo(args.acf, param, args.n_rep, args.seed, args.num_iter)
    meta = res.pop("meta")
    if args.out:
        out = args.out
    else:
        d, out, _tag = cell_paths(args.acf, param, args.n_rep, args.seed)
        os.makedirs(d, exist_ok=True)
    np.savez(out, meta=np.array(meta, dtype=object), **res)  # study_driver.m field schema

    yh = res["Y_min_history"]
    print(f"DONE acf={args.acf} param={param:g} n_rep={args.n_rep} seed={args.seed}  "
          f"final best_y={yh[-1]:.6g}  final true_y_min_est={res['Y_min_est'][-1]:.6g}  "
          f"{meta['runtime']:.1f}s -> {out}")


if __name__ == "__main__":
    main()
