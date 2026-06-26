"""
aleatoric.py -- predicted aleatoric variance r(x) for the heteroscedastic acquisitions.

haei / anpei / rahbo need r(x) = predicted aleatoric (observation-noise) variance at a
candidate x. The MATLAB code (Heter_BO_GF/bayesian_optimizer.m::fit_aleatoric_polymodel)
fits ONE global ridge polynomial:
    log_sigma(W) = theta' . Phi(W),   W = [x1, latent-coords-of-category],
    target = 0.5*log(max(y_var, 1e-12)),  ridge lambda = 1e-3,  degree = 2,
    r(x) = exp(2*log_sigma(x)),
with the latent coordinates giving cross-category structure.

The per-category GP baseline has NO shared latent space (each category is independent),
so the faithful analogue is a PER-CATEGORY degree-2 ridge polynomial in x1 alone -- same
target, degree, ridge, and standardize-then-fit recipe as the MATLAB poly, just fit
separately per level (no information shared across categories, consistent with Method B).
"""
import numpy as np

POLY_DEGREE = 2        # bo_options.poly_degree in study_driver.m
POLY_LAMBDA = 1e-3     # bo_options.poly_lambda in study_driver.m


def _phi(wn, degree):
    """Polynomial design matrix [1, w, w^2, ..., w^degree] for a single feature column.
    Mirrors build_poly_features_local with one feature (no interaction terms)."""
    wn = np.asarray(wn, float).reshape(-1, 1)
    return np.hstack([wn**d for d in range(degree + 1)])      # columns: deg 0..degree


class CategoryAleatoric:
    """Degree-2 ridge log-variance model r(x) for ONE category, in x1 alone."""

    def __init__(self, x1, y_var, degree=POLY_DEGREE, lam=POLY_LAMBDA):
        x1 = np.asarray(x1, float).ravel()
        self.degree = degree
        self.mu = x1.mean()
        sd = x1.std(ddof=1) if x1.size > 1 else 0.0           # sample std, matches MATLAB std(W,0,1)
        self.sd = sd if sd > 0 else 1.0
        wn = (x1 - self.mu) / self.sd
        Phi = _phi(wn, degree)
        log_sigma = 0.5 * np.log(np.maximum(np.asarray(y_var, float).ravel(), 1e-12))
        A = Phi.T @ Phi + lam * np.eye(Phi.shape[1])
        self.theta = np.linalg.solve(A, Phi.T @ log_sigma)

    def predict(self, x_new):
        """Predicted aleatoric VARIANCE r(x) = exp(2*log_sigma(x)), clamped >= 1e-12."""
        wn = (np.asarray(x_new, float).ravel() - self.mu) / self.sd
        log_sigma = _phi(wn, self.degree) @ self.theta
        return np.maximum(np.exp(2 * log_sigma), 1e-12)


class AleatoricModels:
    """Container of per-category aleatoric models, parallel to model.PerCategoryGPs."""

    def __init__(self, models_by_level):
        self.models = dict(models_by_level)

    @classmethod
    def fit(cls, data_by_level, degree=POLY_DEGREE, lam=POLY_LAMBDA):
        """data_by_level: {level: dict(x1=, y_var=, ...)} -> one CategoryAleatoric per level."""
        return cls({lv: CategoryAleatoric(d["x1"], d["y_var"], degree, lam)
                    for lv, d in data_by_level.items()})

    def r(self, level, x_new):
        """Predicted aleatoric variance r(x) for `level` at x_new."""
        return self.models[level].predict(x_new)
