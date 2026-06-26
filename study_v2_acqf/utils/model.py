"""
model.py -- joint categorical GP for the BoTorch-acquisition study (study_v2_acqf).

Same MixedSingleTaskGP + CategoricalKernel as study_v2_cat, but the model is fit on the
NEGATED objective (-y). BoTorch acquisitions/optimizers MAXIMIZE, and our problem is a
MINIMIZATION, so we maximize -f. With the model on -y:
    posterior mean  mu_model = -mu_f   (so argmax mu_model = argmin mu_f)
    best_f          = max(-y_observed) = -min(y_observed)
and the built-in LogEI / UCB / PI work directly in maximize mode (see utils/acqf.py).
"""
import numpy as np
import torch

from botorch.models import MixedSingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

DTYPE = torch.float64


def fit_neg_mixed_gp(data_by_level):
    """Fit ONE MixedSingleTaskGP over all categories on the NEGATED replicate means.
    cat dim = column 1 (code = level-1); only x1 is input-normalized; replicate variance as
    fixed heteroscedastic noise. Returns the fitted model (its mean = -mu_f)."""
    Xs, Ys, Yvs = [], [], []
    for lv, d in data_by_level.items():
        x1 = np.asarray(d["x1"], float).reshape(-1, 1)
        code = np.full_like(x1, float(int(lv) - 1))
        Xs.append(np.hstack([x1, code]))
        Ys.append(-np.asarray(d["y_mean"], float).reshape(-1, 1))     # NEGATE (maximize -f)
        Yvs.append(np.asarray(d["y_var"], float).reshape(-1, 1))
    X = torch.tensor(np.vstack(Xs), dtype=DTYPE)
    Y = torch.tensor(np.vstack(Ys), dtype=DTYPE)
    Yv = torch.tensor(np.vstack(Yvs), dtype=DTYPE).clamp_min(1e-6)
    model = MixedSingleTaskGP(
        train_X=X, train_Y=Y, train_Yvar=Yv, cat_dims=[1],
        input_transform=Normalize(d=2, indices=[0]),
        outcome_transform=Standardize(m=1),
    )
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    model.eval()
    return model
