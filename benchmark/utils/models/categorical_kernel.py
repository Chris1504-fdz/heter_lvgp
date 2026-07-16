"""
categorical_kernel.py -- "categorical_kernel": ONE joint botorch MixedSingleTaskGP +
CategoricalKernel over (x1, category), sharing across levels (the botorch analogue of LVGP's
latent space), with the replicate variance as fixed heteroscedastic noise + the aleatoric poly.
(Adapts study_v2_cat/utils/model.py to the shared Model interface.)
"""
import numpy as np
import torch

from botorch.models import MixedSingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

from .base import BaseModel, AleatoricModels

DTYPE = torch.float64


def _as_2d(a):
    a = np.asarray(a, float)
    return a.reshape(-1, 1) if a.ndim == 1 else a


def _stack(data_by_level):
    Xs, Ys, Yvs = [], [], []
    for lv, d in data_by_level.items():
        Xc = _as_2d(d["X"])                                # (n, d) continuous block
        code = np.full((Xc.shape[0], 1), float(int(lv) - 1))
        Xs.append(np.hstack([Xc, code]))
        Ys.append(np.asarray(d["y_mean"], float).reshape(-1, 1))
        Yvs.append(np.asarray(d["y_var"], float).reshape(-1, 1))
    X = torch.tensor(np.vstack(Xs), dtype=DTYPE)
    Y = torch.tensor(np.vstack(Ys), dtype=DTYPE)
    Yv = torch.tensor(np.vstack(Yvs), dtype=DTYPE).clamp_min(1e-6)
    return X, Y, Yv


def _fit_mixed_gp(data_by_level):
    X, Y, Yv = _stack(data_by_level)
    d = X.shape[1] - 1                                     # continuous dims (last col = category code)
    model = MixedSingleTaskGP(train_X=X, train_Y=Y, train_Yvar=Yv, cat_dims=[d],
                              input_transform=Normalize(d=d + 1, indices=list(range(d))),
                              outcome_transform=Standardize(m=1))
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    model.eval()
    return model


class CategoricalKernelGP(BaseModel):
    ENGINE = "python"
    SUPPORTS = ("ei", "lcb", "pi", "haei", "anpei", "rahbo")

    def __init__(self, model, levels, ale):
        self.model = model
        self.levels = list(levels)
        self._ale = ale

    @classmethod
    def fit(cls, data_by_level, needs_r=True, **_kw):
        model = _fit_mixed_gp(data_by_level)
        ale = AleatoricModels.fit(data_by_level) if needs_r else None
        return cls(model, sorted(int(lv) for lv in data_by_level), ale)

    def predict(self, level, x_new, observation_noise=False):
        x = _as_2d(x_new)
        code = np.full((x.shape[0], 1), float(int(level) - 1))
        X = torch.tensor(np.hstack([x, code]), dtype=DTYPE)
        with torch.no_grad():
            post = self.model.posterior(X, observation_noise=observation_noise)
        return post.mean.numpy().flatten(), post.variance.numpy().flatten()

    def mean_std(self, level, x_new):
        mu, var = self.predict(level, x_new, observation_noise=False)
        return mu, np.maximum(np.sqrt(np.clip(var, 1e-24, None)), 1e-12)
