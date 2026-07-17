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

# One BLAS thread per worker, for BOTH engines. Each worker is already a process, so threaded BLAS
# only fights the other workers. MATLAB's `-singleCompThread` does NOT contain these threads: measured
# 135% mean / 294% peak CPU per MATLAB process, i.e. 8 cells held 21.5 cores instead of ~8.
# Pinning is FREE -- measured identical batch wall (188 s) and per-cell time (121.9 s) for
# heter_LVGP 8-way, and identical python cell times -- while cutting the core footprint ~20%.
# MUST be set before numpy/torch import: OpenBLAS reads the thread count at load time.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
sys.path.insert(0, HERE)
from utils import problems as P, acquisitions
from utils.models import MODELS, BENCHMARK_MODELS, get as get_model

MATLAB = "/data/zhq7531/MATLAB/bin/matlab"
XVFB_BIN = "/data/zhq7531/MATLAB/sys/Xvfb/bin/glnxa64/Xvfb"
XVFB_LIBDIR = "/data/zhq7531/envs/xvfblib/lib"
MATLAB_DRIVER = {"standard_LVGP": "standard_driver", "heter_LVGP": "heter_driver"}
N_REP_LIST = [3, 10]
LAUNCH_GAP_S = 8.0            # min seconds between MATLAB launches (MSH-safe)
# Kill+retry a MATLAB hung in license checkout. Timeout is PER-PROBLEM: measured heter cells run
# ~3-5 h at d<=6 but ~32 h at d=9 under full contention (9-dim hyperopt + 20k-candidate search
# dominate). A flat 8 h would kill every 10-D heter cell mid-run and burn days in retries.
RUN_TIMEOUT_S = 16 * 3600                    # d <= 6 problems; hang guard, NOT a scheduler --
                                             # golinski heter ~4.8h solo stretches to 6-8h under
                                             # multi-user contention, so 8h was killing live cells


def matlab_timeout(function):
    return RUN_TIMEOUT_S if P.get(function).d < 9 else 72 * 3600   # 10-D: 32h solo x contention


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


SCHEMA_VERSION = 2


def _read_meta(path):
    if path.endswith(".npz"):
        with np.load(path, allow_pickle=True) as m:
            m["Y_min_history"]                       # touch a real array: catches truncation
            return dict(m["meta"].item())
    import scipy.io
    d = scipy.io.loadmat(path)
    d["Y_min_history"]
    mm = d["meta"][0, 0]
    return {k: (str(mm[k][0]) if mm[k].dtype.kind in "US" else float(np.ravel(mm[k])[0]))
            for k in mm.dtype.names}


def cell_status(path, num_iter=None):
    """'missing' | 'corrupt' | 'stale' | 'ok'. A cell counts as DONE only if it actually LOADS, was
    written under the CURRENT schema, and used the SAME iteration budget. `os.path.exists` is not enough:
      - a kill/timeout mid-write leaves a truncated file that would be skipped forever ('corrupt');
      - the cell path does NOT encode num_iter, so a 5-iteration `--toy` cell (or an old-schema cell)
        would otherwise be silently accepted in place of a full-budget one ('stale').
    Both 'corrupt' and 'stale' are re-run rather than skipped."""
    if not os.path.exists(path):
        return "missing"
    try:
        meta = _read_meta(path)
    except Exception:
        return "corrupt"
    if int(float(meta.get("schema_version", 1))) < SCHEMA_VERSION:
        return "stale"
    if num_iter is not None and int(float(meta.get("num_iter", -1))) != int(num_iter):
        return "stale"
    return "ok"


def is_loadable(path, num_iter=None):
    return cell_status(path, num_iter) == "ok"


def _run_python(function, model, acf, param, n_rep, seed, num_iter, out):
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(seed)                     # botorch fitting can consume the torch RNG
    from utils import bo
    res = bo.run_bo(P.get(function), get_model(model).cls, acf, param, n_rep, seed, num_iter,
                    model_name=model)
    meta = res.pop("meta")
    # atomic: write to a temp file in the destination dir, verify it loads, then rename over target
    fd, tmp = tempfile.mkstemp(suffix=".npz", dir=os.path.dirname(out)); os.close(fd)
    try:
        with open(tmp, "wb") as fh:
            np.savez(fh, meta=np.array(meta, dtype=object), **res)
        if not is_loadable(tmp):
            raise IOError("wrote an unreadable .npz")
        os.replace(tmp, out)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


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
    from utils import doe_cache
    driver = MATLAB_DRIVER[model]
    doe_file = doe_cache.ensure(function, seed, n_rep)      # SHARED initial design (same file the python models load)
    pstr = "NaN" if param != param else repr(param)
    cmd = [MATLAB, "-nodisplay", "-singleCompThread", "-batch",
           f"{driver}('{function}','{acf}',{pstr},{n_rep},{seed},{num_iter},'{out}','{doe_file}')"]
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
                                    stderr=subprocess.STDOUT, timeout=matlab_timeout(function)).returncode
        except subprocess.TimeoutExpired:
            rc, timed_out = -9, True
        finally:
            shutil.rmtree(pref, ignore_errors=True); shutil.rmtree(tmp, ignore_errors=True)
        if rc == 0 and os.path.exists(out) and is_loadable(out, num_iter):
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
        if is_loadable(out, num_iter):
            return tag, "skip(exists)"
        os.unlink(out)      # truncated / old-schema / wrong-budget leftover -> redo, don't skip
    if get_model(model).engine == "python":
        try:
            _run_python(function, model, acf, param, n_rep, seed, num_iter, out)
            return tag, "ok"
        except Exception as e:
            if os.path.exists(out):
                os.unlink(out)
            return tag, f"FAIL({type(e).__name__}: {e})"
    status = _run_matlab(function, model, acf, param, n_rep, seed, num_iter, out, log)
    if status.startswith("FAIL") and os.path.exists(out):
        os.unlink(out)                          # never leave a partial cell behind
    return tag, status


def build_grid(functions, models, acqs, seeds, num_iter=None):
    """num_iter=None -> each problem uses its own budget from PROBLEM_GRID (resources/init_doe_iter.xlsx:
    50 for the 1-D problems, 200 for the multi-dim ones). Pass an int to override every problem."""
    grid = []
    for fn in functions:
        ni = P.get(fn).num_iter if num_iter is None else num_iter
        for md in models:
            mi = get_model(md)
            cfgs = [(a, p) for (a, p) in acqs if a in mi.supports]      # per-model supported acqs
            for (a, p), nr, s in itertools.product(cfgs, N_REP_LIST, range(1, seeds + 1)):
                grid.append((fn, md, a, p, nr, s, ni))
    return grid


def manifest(grid):
    """Audit the EXPECTED grid against what is on disk. Returns (rows, counts-by-status).
    Without this a corrupt or missing cell just vanishes from the analysis with no non-zero exit."""
    rows = []
    for (fn, md, a, p, nr, s, ni) in grid:
        _d, out, _log, tag = cell_paths(fn, md, a, p, nr, s)
        rows.append(dict(function=fn, model=md, acf=a, param=p, n_rep=nr, seed=s,
                         status=cell_status(out, ni), num_iter=ni, path=out, tag=tag))
    counts = {k: sum(r["status"] == k for r in rows) for k in ("ok", "missing", "stale", "corrupt")}
    return rows, counts


def report_manifest(grid, write_csv=True, max_list=25):
    rows, c = manifest(grid)
    if write_csv:
        import pandas as pd
        path = os.path.join(HERE, "sweep_manifest.csv")
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"manifest -> {path}")
    total = len(rows)
    print(f"\n=== GRID AUDIT: {c['ok']}/{total} ok | {c['missing']} missing | "
          f"{c['stale']} stale | {c['corrupt']} corrupt ===")
    bad_rows = [r for r in rows if r["status"] != "ok"]
    for r in bad_rows[:max_list]:
        print(f"  {r['status'].upper():8s} {r['tag']}")
    if len(bad_rows) > max_list:
        print(f"  ... and {len(bad_rows)-max_list} more (see sweep_manifest.csv)")
    if bad_rows:
        print("!! grid INCOMPLETE -- re-run run.py to fill the gaps "
              "(missing/stale/corrupt cells are all re-run automatically)")
    return c


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
    ap.add_argument("--models", nargs="*", default=list(BENCHMARK_MODELS))   # aux LVGP-validation
                                                                             # models excluded by default
    ap.add_argument("--acqs", nargs="*", default=[a for a, _ in acquisitions.CONFIG_ORDER])
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--num-iter", type=int, default=None,
                    help="override the per-problem budget (default: PROBLEM_GRID / init_doe_iter.xlsx)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--toy", action="store_true",
                    help="smoke test: the FIRST function only x all models x ei x nrep10 x seeds 1-2")
    ap.add_argument("--collect-only", action="store_true")
    ap.add_argument("--verify", action="store_true",
                    help="audit the grid on disk (ok/missing/corrupt) and exit; runs nothing")
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    if args.collect_only:
        df, csv = collect(); print(f"collected {len(df)} rows -> {csv}"); return

    functions = args.functions or P.defined_problems()
    if args.toy:
        functions = functions[:1]
        acqs_cfg = [("ei", float("nan"))]
        toy_iter = args.num_iter or 5                        # toy = smoke test, not the real budget
        grid = [(functions[0], md, "ei", float("nan"), 10, s, toy_iter)
                for md in args.models for s in (1, 2) if "ei" in get_model(md).supports]
    else:
        acqs_cfg = [(a, p) for (a, p) in acquisitions.CONFIG_ORDER if a in args.acqs]
        grid = build_grid(functions, args.models, acqs_cfg, args.seeds, args.num_iter)

    # Schedule SEED-FIRST, most expensive model first within each seed wave:
    #  - seed-first  -> every (problem, seed) is completed across ALL models early, so PAIRED
    #    model comparisons (shared-DOE CRN) are possible wave by wave instead of at the very end;
    #  - cost-descending (LPT) within a wave -> long heter cells start early, shrinking the
    #    end-of-sweep straggler tail. Same cells, same parallelism: zero throughput cost.
    # Queue order (owner's comparison workflow):
    #   1. non-10-D problems before the 10-D block (10-D heter measured ~32 h/cell vs ~5 h at d<=6);
    #   2. ALL nrep=10 before ALL nrep=3 (the 10-replicate comparison is the primary one);
    #   3. PROBLEM-MAJOR within a pass: finish one problem completely -- python models first
    #      (separate_gp, categorical_kernel), then standard_LVGP, then heter_LVGP -- so each
    #      problem's full 4-model comparison lands as a complete unit before the next begins.
    _RANK = {"separate_gp": 0, "categorical_kernel": 1, "standard_LVGP": 2, "heter_LVGP": 3}
    grid.sort(key=lambda c: (P.get(c[0]).d >= 9, c[4] != 10, c[0], _RANK.get(c[1], 9), c[5]))

    if args.verify:
        c = report_manifest(grid)
        sys.exit(0 if c["ok"] == len(grid) else 1)

    needs_matlab = any(get_model(m).engine == "matlab" for m in args.models)
    print(f"{len(grid)} cells | functions {functions} | models {args.models} | "
          f"{args.workers} workers | matlab={needs_matlab}")

    # Pre-generate the shared initial designs SERIALLY. Otherwise the first wave of workers all call
    # doe_cache.ensure on the same keys at once and redundantly rebuild them.
    from utils import doe_cache
    keys = sorted({(fn, s, nr) for (fn, _m, _a, _p, nr, s, _i) in grid})
    for fn, s, nr in keys:
        doe_cache.ensure(fn, s, nr)
    print(f"initial designs ready: {len(keys)} (function, seed, n_rep) combinations")

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
    c = report_manifest(grid)                      # never finish silently on a hole in the grid
    sys.exit(0 if c["ok"] == len(grid) else 1)


if __name__ == "__main__":
    main()
