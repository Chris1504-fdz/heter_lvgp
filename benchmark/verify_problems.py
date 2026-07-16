#!/usr/bin/env python
"""
verify_problems.py -- confirm utils/problems.py (Python) and matlab/problems.m (MATLAB) define the
SAME f_true / sigma for every DEFINED function, so the two engines optimize identical problems.

For each defined function it evaluates f/sigma on a grid of (x1, level) in Python, runs
matlab/eval_problems.m on the same grid, and asserts they agree to ~1e-9. Run this whenever you add
or edit a function.

  python verify_problems.py
"""
import os
import sys
import subprocess
import tempfile
import time
import numpy as np
import scipy.io

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from utils import problems as P

MATLAB = "/data/zhq7531/MATLAB/bin/matlab"
XVFB_BIN = "/data/zhq7531/MATLAB/sys/Xvfb/bin/glnxa64/Xvfb"
XVFB_LIBDIR = "/data/zhq7531/envs/xvfblib/lib"


def _start_xvfb():
    for n in range(120, 160):
        if os.path.exists(f"/tmp/.X11-unix/X{n}"):
            continue
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = XVFB_LIBDIR + ":" + env.get("LD_LIBRARY_PATH", "")
        p = subprocess.Popen([XVFB_BIN, f":{n}", "-screen", "0", "1x1x8", "-nolisten", "tcp"],
                             env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2.0)
        if p.poll() is None:
            return p, f":{n}"
        p.kill()
    raise RuntimeError("no free Xvfb display")


def main():
    funcs = P.defined_problems()
    print(f"verifying {len(funcs)} defined function(s): {funcs}")
    xvfb, disp = _start_xvfb()
    env = dict(os.environ)
    env["DISPLAY"] = disp
    env["LD_LIBRARY_PATH"] = XVFB_LIBDIR + ":" + env.get("LD_LIBRARY_PATH", "")
    env["MATLAB_PREFDIR"] = tempfile.mkdtemp()
    ok = True
    try:
        for name in funcs:
            spec = P.get(name)
            # d==1: the historical 50-pt grid per level. d>1: 200 seeded-random points per level
            # (a dense grid is infeasible; random points test the equations just as strictly).
            if spec.d == 1:
                base = np.linspace(spec.lb, spec.ub, 50).reshape(-1, 1)
            else:
                rng = np.random.default_rng(12345)
                base = rng.uniform(spec.bounds[:, 0], spec.bounds[:, 1], size=(200, spec.d))
            GX, glv = [], []
            for lv in spec.levels:
                GX.append(base); glv.append(np.full(len(base), lv))
            GX = np.vstack(GX); glv = np.concatenate(glv)
            f_py = np.concatenate([np.ravel(spec.f_true_level(GX[glv == lv], int(lv)))
                                   for lv in spec.levels])
            s_py = np.concatenate([np.ravel(spec.sigma_level(GX[glv == lv], int(lv)))
                                   for lv in spec.levels])

            infile = tempfile.mktemp(suffix=".mat"); outfile = tempfile.mktemp(suffix=".mat")
            scipy.io.savemat(infile, {"X": GX, "lv": glv.reshape(-1, 1)})
            cmd = [MATLAB, "-nodisplay", "-batch", f"eval_problems('{name}','{infile}','{outfile}')"]
            rc = subprocess.run(cmd, cwd=os.path.join(HERE, "matlab"), env=env,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
            if rc != 0 or not os.path.exists(outfile):
                print(f"  {name}: MATLAB eval FAILED (rc={rc})"); ok = False; continue
            d = scipy.io.loadmat(outfile)
            f_ml = np.ravel(d["fv"]); s_ml = np.ravel(d["sv"])
            df = np.max(np.abs(f_py - f_ml)); ds = np.max(np.abs(s_py - s_ml))
            status = "OK" if (df < 1e-9 and ds < 1e-9) else "MISMATCH"
            ok = ok and status == "OK"
            print(f"  {name}: max|f_py-f_ml|={df:.2e}  max|sigma_py-sigma_ml|={ds:.2e}  -> {status}")
    finally:
        xvfb.terminate()
    print("\nALL MATCH" if ok else "\nMISMATCH -- fix problems.py / problems.m to agree")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
