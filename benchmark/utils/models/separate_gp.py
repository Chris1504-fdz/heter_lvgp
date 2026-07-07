"""
separate_gp.py -- "Separate_gp": 5 INDEPENDENT per-category botorch SingleTaskGPs (no
cross-category sharing), each with the replicate variance as fixed heteroscedastic noise,
plus the per-category aleatoric poly for the hetero acquisitions.
(Adapts study_v2_gp/utils/model.py to the shared Model interface.)
"""
import numpy as np
import torch

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

from .base import BaseModel, AleatoricModels

DTYPE = torch.float64


def _fit_category_gp(x1, y_mean, y_var):
    X = torch.tensor(np.asarray(x1, float).reshape(-1, 1), dtype=DTYPE)
    Y = torch.tensor(np.asarray(y_mean, float).reshape(-1, 1), dtype=DTYPE)
    Yv = torch.tensor(np.asarray(y_var, float).reshape(-1, 1), dtype=DTYPE).clamp_min(1e-6)
    model = SingleTaskGP(train_X=X, train_Y=Y, train_Yvar=Yv,
                         input_transform=Normalize(d=1), outcome_transform=Standardize(m=1))
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    model.eval()
    return model


class SeparateGP(BaseModel):
    ENGINE = "python"
    SUPPORTS = ("ei", "lcb", "pi", "haei", "anpei", "rahbo")

    def __init__(self, models_by_level, ale):
        self.models = dict(models_by_level)
        self.levels = sorted(self.models)
        self._ale = ale

    @classmethod
    def fit(cls, data_by_level, needs_r=True):
        models = {lv: _fit_category_gp(d["x1"], d["y_mean"], d["y_var"])
                  for lv, d in data_by_level.items()}
        ale = AleatoricModels.fit(data_by_level) if needs_r else None
        return cls(models, ale)

    def predict(self, level, x_new, observation_noise=False):
        Xn = torch.tensor(np.asarray(x_new, float).reshape(-1, 1), dtype=DTYPE)
        with torch.no_grad():
            post = self.models[level].posterior(Xn, observation_noise=observation_noise)
        return post.mean.numpy().flatten(), post.variance.numpy().flatten()

    def mean_std(self, level, x_new):
        mu, var = self.predict(level, x_new, observation_noise=False)
        return mu, np.maximum(np.sqrt(np.clip(var, 1e-24, None)), 1e-12)
