"""
doe_cache.py -- ONE shared initial design per (function, seed, n_rep), generated in Python and loaded
by BOTH engines (common random numbers). This removes the DOE as a confound: for a given seed every
model (separate_gp, categorical_kernel, standard_LVGP, heter_LVGP) starts from the *byte-identical*
initial dataset, so downstream differences are purely model/acquisition.

Design locations = SLHD over the full domain (see notebooks/doe.ipynb); values = n_rep heteroscedastic
noisy replicates. Deterministic in (function, seed, n_rep). The .mat is what MATLAB reads (v7); the
Python models read the same file, so the two engines cannot drift.
"""
import os
import tempfile
import numpy as np
import scipy.io

from . import problems as P

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # benchmark/
CACHE_ROOT = os.path.join(HERE, "doe_cache")


def path(function, seed, n_rep, mode=None):
    """Cache file for one (function, seed, n_rep) under a given DOE mode. The mode is part of the
    FILENAME, so changing problems.DOE_MODE can never silently reuse a design built under the old one."""
    mode = P.DOE_MODE if mode is None else mode
    return os.path.join(CACHE_ROOT, function, f"seed{int(seed):02d}_nrep{int(n_rep):02d}_{mode}.mat")


def _build_mat(function, seed, n_rep, mode):
    """Deterministic full initial dataset -> dict of MATLAB-friendly arrays."""
    d = P.initial_doe(P.get(function), n_rep, seed=seed, mode=mode)   # SLHD + noisy replicates, seeded
    return {
        "X_sample":   np.asarray(d["X_sample"], float),              # (n_tr,2) [x1, level]
        "Y_sample":   np.asarray(d["Y_sample"], float).reshape(-1, 1),
        "Var_sample": np.asarray(d["Var_sample"], float).reshape(-1, 1),
        "Y_rep":      np.asarray(d["Y_rep"], float),                 # (n_tr, n_rep)
        "meta_function": function, "meta_seed": int(seed), "meta_n_rep": int(n_rep),
        "meta_doe_mode": mode, "meta_n_init": int(P.get(function).n_init),
    }


def _stale(p, mode, n_init):
    """Missing, unreadable, or written under a different DOE mode / initial-design size. Belt-and-braces
    on top of the mode-keyed filename -- also catches an n_init change (the spreadsheet budget moving)."""
    if not os.path.exists(p):
        return True
    try:
        m = scipy.io.loadmat(p)
        return (str(np.ravel(m["meta_doe_mode"])[0]).strip() != mode
                or int(np.ravel(m["meta_n_init"])[0]) != int(n_init))
    except Exception:
        return True


def ensure(function, seed, n_rep, mode=None):
    """Write the shared initial-design .mat if missing/stale (atomic) and return its path. Safe under
    the concurrent sweep: several workers building the same key write identical content, rename wins."""
    mode = P.DOE_MODE if mode is None else mode
    p = path(function, seed, n_rep, mode)
    if not _stale(p, mode, P.get(function).n_init):
        return p
    os.makedirs(os.path.dirname(p), exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".mat", dir=os.path.dirname(p)); os.close(fd)
    scipy.io.savemat(tmp, _build_mat(function, seed, n_rep, mode), do_compression=False)
    os.replace(tmp, p)                                               # atomic
    return p


def load(function, seed, n_rep, mode=None):
    """Shared initial dataset as arrays (ensures the .mat exists first, then reads THAT file so the
    Python models see exactly what MATLAB sees)."""
    m = scipy.io.loadmat(ensure(function, seed, n_rep, mode))
    return dict(X_sample=np.asarray(m["X_sample"], float),
                Y_sample=np.ravel(m["Y_sample"]).astype(float),
                Var_sample=np.ravel(m["Var_sample"]).astype(float),
                Y_rep=np.asarray(m["Y_rep"], float))
