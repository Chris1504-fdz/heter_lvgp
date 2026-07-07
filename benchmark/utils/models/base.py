"""
base.py -- the model interface used by the model-agnostic BO loop, plus the shared aleatoric
r(x) (predicted aleatoric VARIANCE) used by haei/anpei/rahbo.

A Python model exposes:
    ENGINE   = "python"
    SUPPORTS = tuple of acquisition names it can run (noise-blind-only models list ei/lcb/pi)
    classmethod fit(data_by_level, needs_r=True) -> instance
    .levels                      : sorted 1-based levels
    .mean_std(level, x)          -> (mu, epistemic_std)   # the (mu, s) the acquisitions use
    .r(level, x)                 -> aleatoric variance     # only if SUPPORTS the hetero acqs

The aleatoric model is the per-category degree-2 ridge log-variance poly ported from
study_v2_gp/utils/aleatoric.py (same target/degree/ridge as Heter_BO_GF's poly).
"""
import numpy as np

POLY_DEGREE = 2
POLY_LAMBDA = 1e-3


def _phi(wn, degree):
    wn = np.asarray(wn, float).reshape(-1, 1)
    return np.hstack([wn ** d for d in range(degree + 1)])


class CategoryAleatoric:
    """Degree-2 ridge log-variance model r(x) for ONE category, in x1 alone."""

    def __init__(self, x1, y_var, degree=POLY_DEGREE, lam=POLY_LAMBDA):
        x1 = np.asarray(x1, float).ravel()
        self.degree = degree
        self.mu = x1.mean()
        sd = x1.std(ddof=1) if x1.size > 1 else 0.0
        self.sd = sd if sd > 0 else 1.0
        wn = (x1 - self.mu) / self.sd
        Phi = _phi(wn, degree)
        log_sigma = 0.5 * np.log(np.maximum(np.asarray(y_var, float).ravel(), 1e-12))
        A = Phi.T @ Phi + lam * np.eye(Phi.shape[1])
        self.theta = np.linalg.solve(A, Phi.T @ log_sigma)

    def predict(self, x_new):
        wn = (np.asarray(x_new, float).ravel() - self.mu) / self.sd
        log_sigma = _phi(wn, self.degree) @ self.theta
        return np.maximum(np.exp(2 * log_sigma), 1e-12)


class AleatoricModels:
    """Container of per-category aleatoric models."""

    def __init__(self, models_by_level):
        self.models = dict(models_by_level)

    @classmethod
    def fit(cls, data_by_level, degree=POLY_DEGREE, lam=POLY_LAMBDA):
        return cls({lv: CategoryAleatoric(d["x1"], d["y_var"], degree, lam)
                    for lv, d in data_by_level.items()})

    def r(self, level, x_new):
        return self.models[level].predict(x_new)


class BaseModel:
    """Mix-in providing r(x) from a fitted AleatoricModels stored on self._ale (or None)."""
    ENGINE = "python"
    SUPPORTS = ("ei", "lcb", "pi", "haei", "anpei", "rahbo")

    def r(self, level, x_new):
        if getattr(self, "_ale", None) is None:
            raise RuntimeError(f"{type(self).__name__} was fit without the aleatoric model (needs_r=False)")
        return self._ale.r(level, x_new)
