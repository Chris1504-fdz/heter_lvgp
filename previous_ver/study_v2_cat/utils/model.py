"""
model.py -- the joint categorical GP (Method C).

ONE botorch MixedSingleTaskGP over (x1, category) with a CategoricalKernel on the
category dimension and the per-point replicate variance as fixed heteroscedastic noise.
Unlike study_v2_gp's per-category baseline (5 independent GPs), this single model SHARES
information across categories through the categorical kernel -- the botorch analogue of
what LVGP does with a learned latent space. Wiring is the "Method C" prototype from
heterockedastic_new_inv/heterosk.ipynb (cells 44, 52: _stack_mixed / _fit_mixed /
botorch_mixed_predict).

Exposes the SAME interface as study_v2_gp/model.PerCategoryGPs (fit / levels / predict /
mean_std), keyed by 1-based level, so utils/bo.py is reused unchanged. Internally the
category is ordinal-encoded code = level-1 in input column 1.
"""
import numpy as np
import torch

from botorch.models import MixedSingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

DTYPE = torch.float64


def _stack(data_by_level):
    """Stack {level: dict(x1, y_mean, y_var)} -> (X[x1, code], Y, Yvar) tensors.
    code = level-1 in column 1 (the categorical dim)."""
    Xs, Ys, Yvs = [], [], []
    for lv, d in data_by_level.items():
        x1 = np.asarray(d["x1"], float).reshape(-1, 1)
        code = np.full_like(x1, float(int(lv) - 1))
        Xs.append(np.hstack([x1, code]))
        Ys.append(np.asarray(d["y_mean"], float).reshape(-1, 1))
        Yvs.append(np.asarray(d["y_var"], float).reshape(-1, 1))
    X = torch.tensor(np.vstack(Xs), dtype=DTYPE)
    Y = torch.tensor(np.vstack(Ys), dtype=DTYPE)
    Yv = torch.tensor(np.vstack(Yvs), dtype=DTYPE).clamp_min(1e-6)
    return X, Y, Yv


def fit_mixed_gp(data_by_level):
    """Fit ONE MixedSingleTaskGP over all categories (notebook `_fit_mixed`).
    Categorical dim = column 1; only x1 (column 0) is input-normalized; fixed
    heteroscedastic noise from the replicate variance."""
    X, Y, Yv = _stack(data_by_level)
    model = MixedSingleTaskGP(
        train_X=X, train_Y=Y, train_Yvar=Yv, cat_dims=[1],
        input_transform=Normalize(d=2, indices=[0]),     # normalize only x1
        outcome_transform=Standardize(m=1),
    )
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    model.eval()
    return model


class MixedCategoryGP:
    """One joint MixedSingleTaskGP shared across all 5 categories (Method C).
    Same interface as study_v2_gp/model.PerCategoryGPs, so utils/bo.py is reused."""

    def __init__(self, model, levels):
        self.model = model
        self.levels = list(levels)

    @classmethod
    def fit(cls, data_by_level):
        model = fit_mixed_gp(data_by_level)
        return cls(model, sorted(int(lv) for lv in data_by_level))

    def predict(self, level, x_new, observation_noise=False):
        """(mean, variance) of the joint GP at (x_new, level). observation_noise=False
        gives the latent (epistemic) variance used for the acquisition's std `s`."""
        x = np.asarray(x_new, float).reshape(-1, 1)
        code = np.full_like(x, float(int(level) - 1))
        X = torch.tensor(np.hstack([x, code]), dtype=DTYPE)
        with torch.no_grad():
            post = self.model.posterior(X, observation_noise=observation_noise)
        return post.mean.numpy().flatten(), post.variance.numpy().flatten()

    def mean_std(self, level, x_new):
        """(mean, epistemic_std) at (x_new, level) -- the (mu, s) the acquisitions use."""
        mu, var = self.predict(level, x_new, observation_noise=False)
        s = np.sqrt(np.clip(var, 1e-24, None))
        return mu, np.maximum(s, 1e-12)
