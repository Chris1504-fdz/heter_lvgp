"""
model.py -- the per-category GP (Method B).

ONE independent botorch SingleTaskGP per categorical level, with the per-point
replicate variance fed in as FIXED heteroscedastic noise (train_Yvar). There is NO
cross-category sharing -- that is the whole point of this baseline. The fit/predict
wiring is taken verbatim from heterockedastic_new_inv/heterosk.ipynb cells 42 & 52
(`fit_botorch_fixed`, `botorch_predict`, `botorch_predict_on`), which were already
validated visually there.

This is NOT the joint categorical model (MixedSingleTaskGP / "Method C") -- that is the
deferred future step and is intentionally absent here.
"""
import numpy as np
import torch

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

DTYPE = torch.float64


def fit_category_gp(x1, y_mean, y_var):
    """Fit one SingleTaskGP on a single category's data (notebook `fit_botorch_fixed`).

    x1, y_mean, y_var : 1-D arrays over that category's observed locations.
    y_var is the per-point replicate (sample) variance -> fixed heteroscedastic noise.
    """
    X = torch.tensor(np.asarray(x1, float).reshape(-1, 1), dtype=DTYPE)
    Y = torch.tensor(np.asarray(y_mean, float).reshape(-1, 1), dtype=DTYPE)
    Yv = torch.tensor(np.asarray(y_var, float).reshape(-1, 1), dtype=DTYPE).clamp_min(1e-6)
    model = SingleTaskGP(
        train_X=X, train_Y=Y, train_Yvar=Yv,
        input_transform=Normalize(d=1),
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    model.eval()
    return model


def predict(model, x_new, observation_noise=False):
    """Posterior (mean, variance) of a per-category GP at x_new (numpy in, numpy out).

    observation_noise=False -> latent (epistemic) variance, used for the acquisition's
    epistemic std `s`. observation_noise=True -> total (adds botorch's flat noise floor);
    used only for coverage/plots, NOT for the heteroscedastic acquisitions (those get the
    aleatoric term r(x) from utils/aleatoric.py instead).
    """
    Xn = torch.tensor(np.asarray(x_new, float).reshape(-1, 1), dtype=DTYPE)
    with torch.no_grad():
        post = model.posterior(Xn, observation_noise=observation_noise)
    return post.mean.numpy().flatten(), post.variance.numpy().flatten()


class PerCategoryGPs:
    """Container for the 5 independent per-category GPs of one BO state.

    Built fresh each BO iteration from the current per-category data. Exposes a uniform
    (mean, epistemic-std) prediction keyed by 1-based level, plus convenience accessors
    used by the acquisition optimizer in bo.py.
    """

    def __init__(self, models_by_level):
        # models_by_level: {level(int, 1-based): fitted SingleTaskGP}
        self.models = dict(models_by_level)
        self.levels = sorted(self.models)

    @classmethod
    def fit(cls, data_by_level):
        """data_by_level: {level: dict(x1=, y_mean=, y_var=)} -> fit one GP per level."""
        models = {lv: fit_category_gp(d["x1"], d["y_mean"], d["y_var"])
                  for lv, d in data_by_level.items()}
        return cls(models)

    def predict(self, level, x_new, observation_noise=False):
        """(mean, variance) for `level` at x_new."""
        return predict(self.models[level], x_new, observation_noise=observation_noise)

    def mean_std(self, level, x_new):
        """(mean, epistemic_std) for `level` at x_new -- the (mu, s) the acquisitions use."""
        mu, var = self.predict(level, x_new, observation_noise=False)
        s = np.sqrt(np.clip(var, 1e-24, None))
        return mu, np.maximum(s, 1e-12)
