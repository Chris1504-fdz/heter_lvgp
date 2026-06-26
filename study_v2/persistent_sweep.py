#!/usr/bin/env python
"""
Run the remaining BO jobs with PERSISTENT MATLAB workers to minimise license
checkouts.

Instead of one `matlab -batch` per run (1080 checkouts), this launches a few
long-lived MATLAB sessions, each looping over a chunk of jobs (run_chunk.m).
License is checked out ONCE per session (N total), held for its lifetime.

  python persistent_sweep.py --workers 4            # finish all pending cells
  python persistent_sweep.py --workers 4 --only anpei rahbo

Resumable: only cells without a .mat are queued.
"""
import argparse, os, sys, time, subprocess, tempfile, shutil
import run_sweep as R   # reuse ACQ_CONFIGS, N_REP_LIST, cell_paths, MATLAB, start_shared_xvfb, HERE

HERE = R.HERE


def pending_jobs(only):
    configs = R.ACQ_CONFIGS
    if only:
        configs = [(a, p) for (a, p) in configs if a in only]
    jobs = []
    for (acf, param) in configs:
        for n_rep in R.N_REP_LIST:
            for seed in range(1, 31):
                d, out, _log, _tag = R.cell_paths(acf, param, n_rep, seed)
                if not os.path.exists(out):
                    os.makedirs(d, exist_ok=True)   # MATLAB save() won't mkdir
                    jobs.append((acf, param, n_rep, seed, out))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4, help="persistent MATLAB sessions")
    ap.add_argument("--num-iter", type=int, default=30)
    ap.add_argument("--stagger", type=float, default=15.0,
                    help="seconds between session startups (avoid checkout burst)")
    ap.add_argument("--max-jobs", type=int, default=None,
                    help="cap total jobs this run (for a quick probe-batch)")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    jobs = pending_jobs(args.only)
    if args.max_jobs:
        jobs = jobs[:args.max_jobs]
    if not jobs:
        print("nothing pending."); return
    N = min(args.workers, len(jobs))
    print(f"{len(jobs)} pending jobs across {N} persistent workers "
          f"(~{len(jobs)//N} each) | num_iter={args.num_iter}")

    # round-robin partition so each worker gets a mix of configs
    chunkdir = os.path.join(HERE, "chunks"); os.makedirs(chunkdir, exist_ok=True)
    chunk_files = []
    for w in range(N):
        cf = os.path.join(chunkdir, f"chunk_{w}.txt")
        with open(cf, "w") as fh:
            for (acf, param, n_rep, seed, out) in jobs[w::N]:
                fh.write(f"{acf},{param},{n_rep},{seed},{out}\n")
        chunk_files.append(cf)

    xvfb_proc, display = R.start_shared_xvfb()
    procs, tmps = [], []
    try:
        for w, cf in enumerate(chunk_files):
            pref = tempfile.mkdtemp(prefix="mlpref_"); tmp = tempfile.mkdtemp(prefix="mltmp_")
            tmps += [pref, tmp]
            env = dict(os.environ)
            env["DISPLAY"] = display
            env["MATLAB_PREFDIR"] = pref
            env["TMPDIR"] = tmp
            log = os.path.join(chunkdir, f"chunk_{w}.log")
            cmd = [R.MATLAB, "-nodisplay", "-singleCompThread", "-batch",
                   f"run_chunk('{cf}', {args.num_iter})"]
            fh = open(log, "w")
            p = subprocess.Popen(cmd, cwd=HERE, env=env, stdout=fh, stderr=subprocess.STDOUT)
            procs.append((p, fh))
            print(f"launched worker {w} (pid {p.pid}) -> {os.path.basename(log)}")
            time.sleep(args.stagger)   # stagger startups so the N checkouts don't burst
        for p, fh in procs:
            p.wait(); fh.close()
    finally:
        xvfb_proc.terminate()
        for d in tmps:
            shutil.rmtree(d, ignore_errors=True)
    done = sum(1 for j in jobs if os.path.exists(j[4]))
    print(f"persistent sweep finished: {done}/{len(jobs)} of the queued jobs now have results")


if __name__ == "__main__":
    main()
