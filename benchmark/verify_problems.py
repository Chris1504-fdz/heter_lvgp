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
            x1 = np.linspace(spec.lb, spec.ub, 50)
            grid_x1, grid_lv = [], []
            for lv in spec.levels:
                grid_x1.append(x1); grid_lv.append(np.full_like(x1, lv))
            gx1 = np.concatenate(grid_x1); glv = np.concatenate(grid_lv)
            f_py = np.array([float(spec.f_true_level(a, int(b))) for a, b in zip(gx1, glv)])
            s_py = np.array([float(spec.sigma_level(a, int(b))) for a, b in zip(gx1, glv)])

            infile = tempfile.mktemp(suffix=".mat"); outfile = tempfile.mktemp(suffix=".mat")
            scipy.io.savemat(infile, {"x1": gx1.reshape(-1, 1), "lv": glv.reshape(-1, 1)})
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
