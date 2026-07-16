"""
lvgp_torch.py -- LVGP surrogates backed by the lab's lvgp-bayes package (Yerramilli et al., the
same Zhang et al. (2019) LVGP the MATLAB engine implements), MAP-fitted in GPyTorch.

Two registry models:
  LVGPTorch      -- "lvgp_torch": homoscedastic, fits the replicate MEAN only, noise-BLIND
                    acquisitions (the pytorch analogue of standard_LVGP; nugget inferred).
  HeterLVGPTorch -- "heter_lvgp_torch": heteroscedastic, replicate variances as FIXED observation
                    noise + the shared aleatoric poly r(x) (the analogue of heter_LVGP).

FIT PROTOCOL: COLD multi-start MAP fit (COLD_RESTARTS scipy restarts) at EVERY iteration --
protocol-matched to every other model in the benchmark (MATLAB LVGP re-multistarts each iteration;
the botorch models refit cold each iteration). Measured: ~10 s per cold fit at n=60, giving a
~4x faster iteration than the MATLAB engine with NO protocol deviation.

WARM_START (default False) exists only as a documented escape hatch if 10-D cold fits ever prove
prohibitive; enabling it is a protocol deviation that must be disclosed AND re-validated -- it makes
the fitting path-dependent, unlike every other model. Do not enable it silently.
Deterministic given torch.manual_seed(seed) (set per cell by run.py).
"""
import numpy as np
import torch

from lvgp_bayes.models import LVGPR
from lvgp_bayes.optim import fit_model_scipy
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

from .base import BaseModel, AleatoricModels, _as_2d

DTYPE = torch.float64
COLD_RESTARTS = 2
RESTART_EVERY = 25
WARM_START = False                    # protocol-matched cold fits; see module docstring


def _stack(data_by_level, bounds):
    """Joint dataset in lvgp-bayes convention: quant cols scaled to the unit cube via the TRUE
    problem bounds (stable across iterations), qual col encoded 0..L-1 appended last."""
    b = np.atleast_2d(np.asarray(bounds, float))
    d = b.shape[0]
    Xs, ys, vs = [], [], []
    for lv in sorted(data_by_level):
        dd = data_by_level[lv]
        X = _as_2d(dd["X"])
        Xs.append(np.hstack([X, np.full((X.shape[0], 1), float(int(lv) - 1))]))
        ys.append(np.asarray(dd["y_mean"], float))
        vs.append(np.asarray(dd["y_var"], float))
    X = np.vstack(Xs); y = np.concatenate(ys); v = np.concatenate(vs)
    Xn = X.copy()
    Xn[:, :d] = (X[:, :d] - b[:, 0]) / (b[:, 1] - b[:, 0])
    return Xn, y, v, d


def _load_matching(model, ref_state):
    """Warm-start: copy every state entry whose shape matches (skips size-changing buffers such as
    the fixed-noise vector, which grows by one point per iteration)."""
    own = model.state_dict()
    for k, val in ref_state.items():
        if k in own and own[k].shape == val.shape:
            own[k] = val
    model.load_state_dict(own)


class _LVGPTorchBase(BaseModel):
    ENGINE = "python"
    HETERO = False

    def __init__(self, model, levels, bounds, ale, n_fits):
        self.model = model
        self.levels = list(levels)
        self._bounds = np.atleast_2d(np.asarray(bounds, float))
        self._ale = ale
        self._n_fits = n_fits                          # fit counter, drives the periodic fresh restart

    @classmethod
    def fit(cls, data_by_level, needs_r=True, bounds=None, warm_from=None, **_kw):
        if bounds is None:
            raise ValueError("lvgp_torch models need bounds= (bo.run_bo passes spec.bounds)")
        Xn, y, v, d = _stack(data_by_level, bounds)
        tx = torch.tensor(Xn, dtype=DTYPE); ty = torch.tensor(y, dtype=DTYPE)
        m = LVGPR(train_x=tx, train_y=ty, qual_index=[d], quant_index=list(range(d)),
                  num_levels_per_var=[len(data_by_level)], lv_dim=2).double()
        if cls.HETERO:                                  # replicate variances as FIXED observation noise
            m.likelihood = FixedNoiseGaussianLikelihood(
                noise=torch.tensor(np.maximum(v, 1e-6), dtype=DTYPE)).double()

        n_fits = 0
        if WARM_START and warm_from is not None and isinstance(warm_from, _LVGPTorchBase):
            n_fits = warm_from._n_fits
            _load_matching(m, warm_from.model.state_dict())
            fresh = 1 if (n_fits % RESTART_EVERY == RESTART_EVERY - 1) else 0
            if fresh:                                   # keep the better of warm-polish vs fresh start
                warm_sd = {k: t.clone() for k, t in m.state_dict().items()}
                fit_model_scipy(m, num_restarts=0)
                warm_mll = _mll_value(m)
                warm_fit_sd = {k: t.clone() for k, t in m.state_dict().items()}
                _load_matching(m, warm_sd)
                fit_model_scipy(m, num_restarts=1)
                if _mll_value(m) < warm_mll:
                    _load_matching(m, warm_fit_sd)
            else:
                fit_model_scipy(m, num_restarts=0)
        else:
            fit_model_scipy(m, num_restarts=COLD_RESTARTS)
        m.eval()
        ale = AleatoricModels.fit(data_by_level) if needs_r else None
        return cls(m, sorted(int(lv) for lv in data_by_level), bounds, ale, n_fits + 1)

    def _encode(self, level, x_new):
        X = _as_2d(x_new)
        b = self._bounds
        Xn = (X - b[:, 0]) / (b[:, 1] - b[:, 0])
        code = np.full((Xn.shape[0], 1), float(int(level) - 1))
        return torch.tensor(np.hstack([Xn, code]), dtype=DTYPE)

    def predict(self, level, x_new, observation_noise=False):
        with torch.no_grad():
            post = self.model(self._encode(level, x_new))          # latent (noise-free) posterior
        return post.mean.numpy().ravel(), post.variance.numpy().ravel()

    def mean_std(self, level, x_new):
        mu, var = self.predict(level, x_new)
        return mu, np.maximum(np.sqrt(np.clip(var, 1e-24, None)), 1e-12)


def _mll_value(model):
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    model.train()
    with torch.no_grad():
        out = model(*model.train_inputs)
        val = float(mll(out, model.train_targets))
    model.eval()
    return val


class LVGPTorch(_LVGPTorchBase):
    """Homoscedastic LVGP on the replicate mean (noise-blind acquisitions only)."""
    SUPPORTS = ("ei", "lcb", "pi")
    HETERO = False


class HeterLVGPTorch(_LVGPTorchBase):
    """Heteroscedastic LVGP: replicate variances as fixed noise + aleatoric poly r(x)."""
    SUPPORTS = ("ei", "lcb", "pi", "haei", "anpei", "rahbo")
    HETERO = True
