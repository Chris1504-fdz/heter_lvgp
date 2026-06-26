#!/usr/bin/env python
"""
Parameter study for the heteroscedastic LVGP Bayesian optimization code.

Sweeps 12 acquisition-function configs x n_rep in {3,5,10} x N seeds, by
launching one isolated `matlab -batch study_driver(...)` per cell across a
local process pool, then collecting every Y_min_history into a tidy CSV.

The .m files are NOT modified; study_driver.m only *calls* them.
"""
import argparse, subprocess, itertools, os, sys, tempfile, shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
import scipy.io
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
MATLAB = "/data/zhq7531/MATLAB/bin/matlab"

# Shared headless display: MATLAB R2026a auto-spawns its own Xvfb per process
# otherwise (collides under concurrency). We run ONE Xvfb and point all runs at
# it. Its bundled Xvfb needs libtirpc (installed in an isolated conda env).
XVFB_BIN = "/data/zhq7531/MATLAB/sys/Xvfb/bin/glnxa64/Xvfb"
XVFB_LIBDIR = "/data/zhq7531/envs/xvfblib/lib"


def start_shared_xvfb():
    """Launch one Xvfb on the first free display; return (proc, ':N')."""
    import time
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

# The 12 chosen acquisition configs (acf, knob_value). knob is NaN when unused.
ACQ_CONFIGS = [
    ("lcb",  float("nan")),
    ("pi",   float("nan")),
    ("ei",   float("nan")),
    ("haei", 0.5), ("haei", 1.0), ("haei", 5.0),
    ("anpei", 0.2), ("anpei", 0.5), ("anpei", 0.8),
    ("rahbo", 0.5), ("rahbo", 1.0), ("rahbo", 5.0),
]
N_REP_LIST = [3, 5, 10]

# Small representative subset for a quick visualization run (covers each acf family).
TOY_CONFIGS = [
    ("ei", float("nan")),
    ("lcb", float("nan")),
    ("haei", 1.0),
    ("anpei", 0.5),
    ("rahbo", 1.0),
]


# knob letter per acquisition family, for human-readable folder names
_KNOB = {"haei": "g", "rahbo": "a", "anpei": "b"}


def acf_tag(acf, param):
    """Folder name for an acquisition config, e.g. 'ei', 'haei_g0.5', 'rahbo_a1'."""
    if param != param:                     # NaN -> no knob (ei/lcb/pi)
        return acf
    return f"{acf}_{_KNOB.get(acf, 'p')}{param:g}"


def cell_paths(acf, param, n_rep, seed):
    """Nested layout: results/<acf_tag>/nrep<NN>/seed<NN>.{mat,log} + a print tag."""
    d = os.path.join(RESULTS, acf_tag(acf, param), f"nrep{n_rep:02d}")
    base = f"seed{seed:02d}"
    tag = f"{acf_tag(acf, param)}/nrep{n_rep:02d}/{base}"
    return d, os.path.join(d, base + ".mat"), os.path.join(d, base + ".log"), tag


def run_one(args):
    acf, param, n_rep, seed, num_iter = args
    d, out, log, tag = cell_paths(acf, param, n_rep, seed)
    os.makedirs(d, exist_ok=True)
    if os.path.exists(out):
        return tag, "skip(exists)"
    pstr = "NaN" if param != param else repr(param)
    # Keep the JVM (study_driver needs parallel.Settings to disable the auto
    # pool). One process per run; outer parallelism is handled here.
    cmd = [MATLAB, "-nodisplay", "-singleCompThread", "-batch",
           f"study_driver('{acf}', {pstr}, {n_rep}, {seed}, {num_iter}, '{out}')"]
    env = dict(os.environ)
    env["DISPLAY"] = os.environ.get("STUDY_DISPLAY", "")   # shared Xvfb
    # Online (MHLM) licensing rejects bursts of concurrent launches with
    # error 5001. Retry those (and only those) a few times with backoff so the
    # sweep self-heals; genuine code failures (e.g. anpei) are not retried.
    import time as _t, random as _r
    for attempt in range(6):
        pref = tempfile.mkdtemp(prefix="mlpref_"); tmp = tempfile.mkdtemp(prefix="mltmp_")
        env["MATLAB_PREFDIR"] = pref; env["TMPDIR"] = tmp
        try:
            with open(log, "w") as fh:
                rc = subprocess.run(cmd, cwd=HERE, env=env,
                                    stdout=fh, stderr=subprocess.STDOUT).returncode
        finally:
            shutil.rmtree(pref, ignore_errors=True); shutil.rmtree(tmp, ignore_errors=True)
        if rc == 0 and os.path.exists(out):
            return tag, ("ok" if attempt == 0 else f"ok(retry {attempt})")
        try:
            is_5001 = "5001" in open(log).read()
        except Exception:
            is_5001 = False
        if not is_5001:
            return tag, f"FAIL(rc={rc})"          # real failure (e.g. anpei bug)
        _t.sleep(20 + 10*attempt + _r.uniform(0, 8))   # license blip -> back off, retry
    return tag, "FAIL(5001 x6)"


def collect():
    import glob
    rows = []
    for f in glob.glob(os.path.join(RESULTS, "**", "*.mat"), recursive=True):
        m = scipy.io.loadmat(f)
        if "Y_min_history" not in m:
            continue
        y = np.ravel(m["Y_min_history"]).astype(float)
        meta = m["meta"][0, 0]
        acf = str(meta["acf"][0])
        param = float(np.ravel(meta["acf_param"])[0])
        n_rep = int(np.ravel(meta["n_rep"])[0])
        seed = int(np.ravel(meta["seed"])[0])
        runtime = float(np.ravel(meta["runtime"])[0])
        for it, val in enumerate(y, start=1):
            rows.append(dict(acf=acf, param=param, n_rep=n_rep, seed=seed,
                             iter=it, best_y=val, runtime=runtime))
    df = pd.DataFrame(rows).sort_values(["acf", "param", "n_rep", "seed", "iter"])
    csv = os.path.join(HERE, "sweep_results.csv")
    df.to_csv(csv, index=False)
    return df, csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20, help="number of seeds per condition")
    ap.add_argument("--num-iter", type=int, default=30, help="BO iterations per run")
    ap.add_argument("--workers", type=int, default=8, help="concurrent MATLAB processes")
    ap.add_argument("--collect-only", action="store_true", help="just rebuild the CSV")
    ap.add_argument("--toy", action="store_true",
                    help="quick visualization run: 5-config subset, small grid")
    ap.add_argument("--only", nargs="*", default=None,
                    help="restrict to these acquisition names, e.g. --only rahbo")
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)

    if args.collect_only:
        df, csv = collect()
        print(f"Collected {len(df)} rows -> {csv}")
        return

    configs = TOY_CONFIGS if args.toy else ACQ_CONFIGS
    if args.only:
        configs = [(a, p) for (a, p) in configs if a in args.only]
    grid = [(acf, param, n_rep, seed, args.num_iter)
            for (acf, param), n_rep, seed
            in itertools.product(configs, N_REP_LIST, range(1, args.seeds + 1))]
    print(f"{len(grid)} runs | {args.workers} workers | num_iter={args.num_iter}")

    xvfb_proc, display = start_shared_xvfb()
    os.environ["STUDY_DISPLAY"] = display      # inherited by pool workers (fork)
    done = 0
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(run_one, g): g for g in grid}
            for fut in as_completed(futs):
                tag, status = fut.result()
                done += 1
                print(f"[{done}/{len(grid)}] {tag}: {status}", flush=True)
    finally:
        xvfb_proc.terminate()

    df, csv = collect()
    print(f"\nCollected {len(df)} rows -> {csv}")


if __name__ == "__main__":
    main()
