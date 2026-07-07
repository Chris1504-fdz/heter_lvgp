#!/usr/bin/env python
"""
run.py -- dispatcher + resumable sweep for the 9-function x 4-model mixed-variable BO grid.

Each cell = (function, model, acquisition, n_rep, seed). The model's registry entry says which
ENGINE runs it: python models call utils.bo.run_bo -> .npz; matlab (LVGP) models launch the
parameterized MATLAB driver -> .mat. Both write results/<function>/<model>/<acq>/nrep<NN>/seed<NN>.
MATLAB launches are throttled (>=LAUNCH_GAP_S apart) + timed out, so a burst can't wedge the MSH.

  python run.py --toy                              # small probe (branin x 4 models x ei x nrep10 x seeds 1-2)
  python run.py --functions branin_hetero --models separate_gp categorical_kernel --seeds 30 --workers 8
  python run.py --collect-only
"""
import argparse
import itertools
import os
import sys
import subprocess
import tempfile
import shutil
import time
import random
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
sys.path.insert(0, HERE)
from utils import problems as P, acquisitions
from utils.models import MODELS, get as get_model

MATLAB = "/data/zhq7531/MATLAB/bin/matlab"
XVFB_BIN = "/data/zhq7531/MATLAB/sys/Xvfb/bin/glnxa64/Xvfb"
XVFB_LIBDIR = "/data/zhq7531/envs/xvfblib/lib"
MATLAB_DRIVER = {"standard_LVGP": "standard_driver", "heter_LVGP": "heter_driver"}
N_REP_LIST = [3, 5, 10]
LAUNCH_GAP_S = 8.0            # min seconds between MATLAB launches (MSH-safe)
RUN_TIMEOUT_S = 1800         # kill+retry a MATLAB hung in license checkout


def start_shared_xvfb():
    for n in range(90, 130):
        if os.path.exists(f"/tmp/.X11-unix/X{n}"):
            continue
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = XVFB_LIBDIR + ":" + env.get("LD_LIBRARY_PATH", "")
        p = subprocess.Popen([XVFB_BIN, f":{n}", "-screen", "0", "1x1x8", "-nolisten", "tcp"],
                             env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2.0)
        if p.poll() is None and os.path.exists(f"/tmp/.X11-unix/X{n}"):
            print(f"shared Xvfb on :{n} (pid {p.pid})")
            return p, f":{n}"
        p.kill()
    raise RuntimeError("could not start a shared Xvfb")


def cell_paths(function, model, acf, param, n_rep, seed):
    tag = acquisitions.acf_tag(acf, param)
    d = os.path.join(RESULTS, function, model, tag, f"nrep{n_rep:02d}")
    base = f"seed{seed:02d}"
    ext = ".mat" if get_model(model).engine == "matlab" else ".npz"
    return (d, os.path.join(d, base + ext), os.path.join(d, base + ".log"),
            f"{function}/{model}/{tag}/nrep{n_rep:02d}/{base}")


def _run_python(function, model, acf, param, n_rep, seed, num_iter, out):
    import torch
    torch.set_num_threads(1)
    from utils import bo
    res = bo.run_bo(P.get(function), get_model(model).cls, acf, param, n_rep, seed, num_iter,
                    model_name=model)
    meta = res.pop("meta")
    np.savez(out, meta=np.array(meta, dtype=object), **res)


# ---- MATLAB launch throttle (shared across workers) ----
def _pool_init(lock, last):
    global _LK, _LAST
    _LK, _LAST = lock, last


def _throttle():
    lk = globals().get("_LK"); last = globals().get("_LAST")
    if lk is None:
        return
    with lk:
        wait = last.value + LAUNCH_GAP_S - time.time()
        if wait > 0:
            time.sleep(wait)
        last.value = time.time()


def _run_matlab(function, model, acf, param, n_rep, seed, num_iter, out, log):
    driver = MATLAB_DRIVER[model]
    pstr = "NaN" if param != param else repr(param)
    cmd = [MATLAB, "-nodisplay", "-singleCompThread", "-batch",
           f"{driver}('{function}','{acf}',{pstr},{n_rep},{seed},{num_iter},'{out}')"]
    env = dict(os.environ)
    env["DISPLAY"] = os.environ.get("BENCH_DISPLAY", "")
    env["LD_LIBRARY_PATH"] = XVFB_LIBDIR + ":" + env.get("LD_LIBRARY_PATH", "")
    for attempt in range(6):
        pref = tempfile.mkdtemp(prefix="mlpref_"); tmp = tempfile.mkdtemp(prefix="mltmp_")
        env["MATLAB_PREFDIR"] = pref; env["TMPDIR"] = tmp
        _throttle()
        timed_out = False
        try:
            with open(log, "w") as fh:
                rc = subprocess.run(cmd, cwd=os.path.join(HERE, "matlab"), env=env, stdout=fh,
                                    stderr=subprocess.STDOUT, timeout=RUN_TIMEOUT_S).returncode
        except subprocess.TimeoutExpired:
            rc, timed_out = -9, True
        finally:
            shutil.rmtree(pref, ignore_errors=True); shutil.rmtree(tmp, ignore_errors=True)
        if rc == 0 and os.path.exists(out):
            return "ok" if attempt == 0 else f"ok(retry {attempt})"
        try:
            is_5001 = "5001" in open(log).read()
        except Exception:
            is_5001 = False
        if not (is_5001 or timed_out):
            return f"FAIL(rc={rc})"
        _throttle()
        time.sleep(20 + 10 * attempt + random.uniform(0, 8))
    return "FAIL(retry x6)"


def run_one(args):
    function, model, acf, param, n_rep, seed, num_iter = args
    d, out, log, tag = cell_paths(function, model, acf, param, n_rep, seed)
    os.makedirs(d, exist_ok=True)
    if os.path.exists(out):
        return tag, "skip(exists)"
    if get_model(model).engine == "python":
        try:
            _run_python(function, model, acf, param, n_rep, seed, num_iter, out)
            return tag, "ok"
        except Exception as e:
            return tag, f"FAIL({type(e).__name__}: {e})"
    return tag, _run_matlab(function, model, acf, param, n_rep, seed, num_iter, out, log)


def build_grid(functions, models, acqs, seeds, num_iter):
    grid = []
    for fn in functions:
        for md in models:
            mi = get_model(md)
            cfgs = [(a, p) for (a, p) in acqs if a in mi.supports]      # per-model supported acqs
            for (a, p), nr, s in itertools.product(cfgs, N_REP_LIST, range(1, seeds + 1)):
                grid.append((fn, md, a, p, nr, s, num_iter))
    return grid


def collect():
    import glob
    import scipy.io
    rows = []
    for f in glob.glob(os.path.join(RESULTS, "**", "*.npz"), recursive=True) + \
             glob.glob(os.path.join(RESULTS, "**", "*.mat"), recursive=True):
        try:
            if f.endswith(".npz"):
                m = np.load(f, allow_pickle=True); meta = m["meta"].item()
            else:
                d = scipy.io.loadmat(f); mm = d["meta"][0, 0]
                meta = {k: (str(mm[k][0]) if mm[k].dtype.kind in "US" else float(np.ravel(mm[k])[0]))
                        for k in mm.dtype.names}
            y = np.ravel((m["Y_min_history"] if f.endswith(".npz") else d["Y_min_history"])).astype(float)
            rows.append(dict(problem=meta.get("problem"), model=meta.get("model"), acf=meta.get("acf"),
                             param=meta.get("acf_param"), n_rep=int(float(meta.get("n_rep"))),
                             seed=int(float(meta.get("seed"))), final_best=float(y[-1]),
                             runtime=float(meta.get("runtime", 0))))
        except Exception as e:
            print(f"  skip {f}: {e}")
    import pandas as pd
    df = pd.DataFrame(rows)
    csv = os.path.join(HERE, "sweep_results.csv")
    df.to_csv(csv, index=False)
    return df, csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--functions", nargs="*", default=None, help="default: all DEFINED problems")
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--acqs", nargs="*", default=[a for a, _ in acquisitions.CONFIG_ORDER])
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--num-iter", type=int, default=30)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--toy", action="store_true", help="branin x all models x ei x nrep10 x seeds 1-2")
    ap.add_argument("--collect-only", action="store_true")
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    if args.collect_only:
        df, csv = collect(); print(f"collected {len(df)} rows -> {csv}"); return

    functions = args.functions or P.defined_problems()
    if args.toy:
        functions = functions[:1]
        acqs_cfg = [("ei", float("nan"))]
        grid = [(functions[0], md, "ei", float("nan"), 10, s, args.num_iter)
                for md in args.models for s in (1, 2) if "ei" in get_model(md).supports]
    else:
        acqs_cfg = [(a, p) for (a, p) in acquisitions.CONFIG_ORDER if a in args.acqs]
        grid = build_grid(functions, args.models, acqs_cfg, args.seeds, args.num_iter)

    needs_matlab = any(get_model(m).engine == "matlab" for m in args.models)
    print(f"{len(grid)} cells | functions {functions} | models {args.models} | "
          f"{args.workers} workers | matlab={needs_matlab}")

    xvfb_proc = None
    if needs_matlab:
        xvfb_proc, disp = start_shared_xvfb()
        os.environ["BENCH_DISPLAY"] = disp
    lock = mp.Lock(); last = mp.Value("d", 0.0)
    done = 0
    try:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_pool_init,
                                 initargs=(lock, last)) as ex:
            futs = {ex.submit(run_one, g): g for g in grid}
            for fut in as_completed(futs):
                tag, status = fut.result(); done += 1
                print(f"[{done}/{len(grid)}] {tag}: {status}", flush=True)
    finally:
        if xvfb_proc is not None:
            xvfb_proc.terminate()
    df, csv = collect()
    print(f"\ncollected {len(df)} rows -> {csv}")


if __name__ == "__main__":
    main()
