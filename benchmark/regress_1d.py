#!/usr/bin/env python
"""
regress_1d.py -- PHASE 2a acceptance test.

Re-runs a sample of the already-computed 1-D cells and diffs them BIT-FOR-BIT against the stored
results/. The d-dim generalization must reduce EXACTLY to the current code when d == 1, so this must
keep passing through the whole refactor. Run it before touching anything (to prove the pipeline is
reproducible at all) and after every step.

    python regress_1d.py            # python-engine cells
    python regress_1d.py --matlab   # also re-run the MATLAB cells (slow)
"""
import argparse
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from utils import problems as P                              # noqa: E402
from utils.models import get as get_model                    # noqa: E402
import run as RUN                                            # noqa: E402

# a spread of cells: both python models, noise-blind + noise-aware acqs, both n_rep, several seeds
PY_CELLS = [
    ("branin_hetero", "separate_gp", "ei", float("nan"), 10, 1),
    ("branin_hetero", "separate_gp", "rahbo", 0.5, 3, 7),
    ("branin_hetero", "categorical_kernel", "anpei", 0.8, 10, 2),
    ("sixhump_camel", "categorical_kernel", "lcb", float("nan"), 3, 5),
    ("griewank_2d", "separate_gp", "haei", 0.5, 10, 3),
    ("ackley_2d", "categorical_kernel", "pi", float("nan"), 10, 4),
]
ML_CELLS = [
    ("branin_hetero", "heter_LVGP", "ei", float("nan"), 10, 1),
    ("branin_hetero", "standard_LVGP", "lcb", float("nan"), 3, 2),
]
ARRAYS = ["X_sampled", "Y_sampled", "Y_var_sampled", "Y_min_history", "X_min_est",
          "X_init", "Y_init", "Y_rep_init", "Y_rep_sampled", "f_true_sampled", "sigma_true_sampled"]


def _load(path):
    if path.endswith(".npz"):
        with np.load(path, allow_pickle=True) as m:
            return {k: np.asarray(m[k]) for k in m.files if k != "meta"}
    import scipy.io
    d = scipy.io.loadmat(path)
    return {k: np.asarray(v) for k, v in d.items() if not k.startswith("__") and k != "meta"}


def _diff(a, b):
    """Max abs difference between two same-named arrays (NaN-safe); None if shapes differ."""
    a = np.asarray(a, float).squeeze(); b = np.asarray(b, float).squeeze()
    if a.shape != b.shape:
        return None
    if a.size == 0:
        return 0.0
    d = np.abs(a - b)
    return float(np.nanmax(d)) if d.size else 0.0


def run_python_cell(fn, model, acf, param, n_rep, seed, num_iter):
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(seed)                                   # exactly as run.py::_run_python
    from utils import bo
    res = bo.run_bo(P.get(fn), get_model(model).cls, acf, param, n_rep, seed, num_iter,
                    model_name=model)
    res.pop("meta", None)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matlab", action="store_true")
    ap.add_argument("--tol", type=float, default=0.0, help="0 = require bit-exact")
    args = ap.parse_args()

    n_ok = n_bad = n_skip = 0
    print(f"{'cell':58s}{'result':>12s}  worst |Δ|")
    print("-" * 88)

    for (fn, model, acf, param, n_rep, seed) in PY_CELLS:
        _d, out, _log, tag = RUN.cell_paths(fn, model, acf, param, n_rep, seed)
        if not os.path.exists(out):
            print(f"{tag:58s}{'SKIP(absent)':>12s}"); n_skip += 1; continue
        stored = _load(out)
        fresh = run_python_cell(fn, model, acf, param, n_rep, seed, P.get(fn).num_iter)
        worst, bad = 0.0, []
        for k in ARRAYS:
            if k not in stored or k not in fresh:
                continue
            d = _diff(stored[k], fresh[k])
            if d is None:
                bad.append(f"{k}:shape"); continue
            worst = max(worst, d)
            if d > args.tol:
                bad.append(f"{k}:{d:.2e}")
        ok = not bad
        n_ok += ok; n_bad += (not ok)
        print(f"{tag:58s}{'PASS' if ok else 'FAIL':>12s}  {worst:.2e}"
              + ("" if ok else "   " + ", ".join(bad[:4])))

    if args.matlab:
        import subprocess
        import scipy.io
        from utils import doe_cache
        xvfb, disp = RUN.start_shared_xvfb()
        try:
            for (fn, model, acf, param, n_rep, seed) in ML_CELLS:
                _d, out, _log, tag = RUN.cell_paths(fn, model, acf, param, n_rep, seed)
                if not os.path.exists(out):
                    print(f"{tag:58s}{'SKIP(absent)':>12s}"); n_skip += 1; continue
                tmp = tempfile.mkdtemp()
                dst = os.path.join(tmp, "fresh.mat")
                doe = doe_cache.ensure(fn, seed, n_rep)
                drv = RUN.MATLAB_DRIVER[model]
                pstr = "NaN" if param != param else repr(param)
                env = dict(os.environ); env["DISPLAY"] = disp
                env["LD_LIBRARY_PATH"] = RUN.XVFB_LIBDIR + ":" + env.get("LD_LIBRARY_PATH", "")
                subprocess.run([RUN.MATLAB, "-nodisplay", "-singleCompThread", "-batch",
                                f"{drv}('{fn}','{acf}',{pstr},{n_rep},{seed},"
                                f"{P.get(fn).num_iter},'{dst}','{doe}')"],
                               cwd=os.path.join(HERE, "matlab"), env=env,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=RUN.RUN_TIMEOUT_S)
                if not os.path.exists(dst):
                    print(f"{tag:58s}{'FAIL(no out)':>12s}"); n_bad += 1; continue
                stored, fresh = _load(out), _load(dst)
                worst, bad = 0.0, []
                for k in ARRAYS:
                    if k not in stored or k not in fresh:
                        continue
                    d = _diff(stored[k], fresh[k])
                    if d is None:
                        bad.append(f"{k}:shape"); continue
                    worst = max(worst, d)
                    if d > args.tol:
                        bad.append(f"{k}:{d:.2e}")
                ok = not bad
                n_ok += ok; n_bad += (not ok)
                print(f"{tag:58s}{'PASS' if ok else 'FAIL':>12s}  {worst:.2e}"
                      + ("" if ok else "   " + ", ".join(bad[:4])))
        finally:
            xvfb.terminate()

    print("-" * 88)
    print(f"PASS {n_ok} | FAIL {n_bad} | SKIP {n_skip}   (tol={args.tol})")
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
