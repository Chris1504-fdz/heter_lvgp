"""
model.py -- per-category heteroscedastic GP (HGPR; Ozbayram, Olivier, Graham-Brady, CMAME 2024).

One HGPR per category, each LEARNING its own polynomial noise model
    sigma^2(x) = (k_al * exp(sum_{i=1..d} theta_i x^i))^2
jointly with the GP via a MAP loss (Gaussian prior on theta, log-normal priors on l, k_al).
This REPLACES study_v2_gp's (SingleTaskGP + separately-fit aleatoric poly): here the aleatoric
r(x) used by the hetero-aware acquisitions IS the HGPR's learned sigma^2(x), so the noise model
and the surrogate are trained together (the point of this study).

Hyperparameters are the config validated in heterosk.ipynb Experiment 1 (mean MAE 9.11 on the
Branin levels): poly_degree=2 (log-sigma is exactly quadratic in x1 here -> well specified),
mu_l=-1.5, lambda_l_sq=1.0.  The tight-prior variant (mu_l=-2.5, lambda_l_sq=0.05) overfits the
length scale (l ~ 0.08) and scores 14.3 -- deliberately NOT used.

x is normalized to [0,1] over the FIXED domain [LB,UB] (keeps the length-scale prior meaningful
and lets us warm-start across BO iterations); y is normalized to [0,1] over its data range.
Each BO iteration changes only one category, so bo.py cold-fits all 5 once and then warm-refits
only the changed category (EPOCHS_WARM) -- ~0.6 s vs ~8 s cold.
"""
import numpy as np
import torch
import torch.nn as nn

from . import problem

DTYPE = torch.float64
POLY_DEGREE = 2
MU_L, LAMBDA_L_SQ = -1.5, 1.0
MU_K_AL, LAMBDA_K_AL_SQ = -1.0, 1.0
LAMBDA_THETA_SQ = 1.0
LR = 0.01
EPOCHS_COLD = 2000
EPOCHS_WARM = 300
JITTER = 1e-5


class HGPR1D(nn.Module):
    """1-D heteroscedastic GP with a polynomial noise model (Ozbayram et al. 2024, Sec. 2.2)."""

    def __init__(self, poly_degree=POLY_DEGREE):
        super().__init__()
        self.poly_degree = poly_degree
        self.l = nn.Parameter(torch.tensor([0.1], dtype=DTYPE))
        self.k_al = nn.Parameter(torch.tensor(1.0, dtype=DTYPE))
        self.theta = nn.Parameter(torch.ones(poly_degree, dtype=DTYPE))

    def kernel(self, X1, X2):
        l = torch.nn.functional.softplus(self.l)
        return torch.exp(-0.5 * torch.cdist(X1 / l, X2 / l) ** 2)

    def sigma2(self, X):
        k = torch.nn.functional.softplus(self.k_al)
        poly = (sum(self.theta[d - 1] * X ** d for d in range(1, self.poly_degree + 1))
                if self.poly_degree > 0 else torch.zeros_like(X))
        return (k * torch.exp(poly)).flatten() ** 2

    def map_loss(self, X, y):
        N = X.shape[0]
        K = self.kernel(X, X) + torch.diag(self.sigma2(X)) + JITTER * torch.eye(N, dtype=DTYPE)
        L = torch.linalg.cholesky(K)
        a = torch.cholesky_solve(y, L)
        nll = 0.5 * (y.T @ a).squeeze() + torch.log(torch.diag(L)).sum() + 0.5 * N * np.log(2 * np.pi)
        l_p = torch.nn.functional.softplus(self.l)
        k_p = torch.nn.functional.softplus(self.k_al)
        R_theta = -0.5 / LAMBDA_THETA_SQ * (self.theta ** 2).sum()
        R_k = (-torch.log(k_p * np.sqrt(LAMBDA_K_AL_SQ * 2 * np.pi))
               - (torch.log(k_p) - MU_K_AL) ** 2 / (2 * LAMBDA_K_AL_SQ))
        R_l = (-torch.log(l_p * np.sqrt(LAMBDA_L_SQ * 2 * np.pi))
               - (torch.log(l_p) - MU_L) ** 2 / (2 * LAMBDA_L_SQ)).sum()
        return nll - R_theta - R_k - R_l


def _flatten(d):
    """Per-category data dict -> (x_raw, y_raw) over ALL replicates (HGPR learns sigma^2 from scatter)."""
    x = np.concatenate([np.full(len(yr), float(xx)) for xx, yr in zip(d["x1"], d["y_rep"])])
    y = np.concatenate([np.asarray(yr, float) for yr in d["y_rep"]])
    return x, y


class _LevelHGPR:
    """One category's HGPR + its normalization + a cached Cholesky factor for fast prediction."""

    def __init__(self):
        self.model = HGPR1D()
        self.X_min, self.rx = problem.LB, problem.UB - problem.LB   # FIXED-domain x normalization

    def fit(self, x_raw, y_raw, epochs):
        """(Re)fit on raw replicates. Warm-starts from current params (refit_level passes few epochs)."""
        self.y_min = float(np.min(y_raw))
        self.ry = float(np.max(y_raw) - np.min(y_raw)) or 1.0
        self.Xt = torch.tensor(((x_raw - self.X_min) / self.rx).reshape(-1, 1), dtype=DTYPE)
        self.yt = torch.tensor(((y_raw - self.y_min) / self.ry).reshape(-1, 1), dtype=DTYPE)
        opt = torch.optim.Adam(self.model.parameters(), lr=LR)
        for _ in range(epochs):
            opt.zero_grad()
            loss = self.model.map_loss(self.Xt, self.yt)
            loss.backward()
            opt.step()
        self._cache()

    def _cache(self):
        with torch.no_grad():
            N = self.Xt.shape[0]
            K = (self.model.kernel(self.Xt, self.Xt) + torch.diag(self.model.sigma2(self.Xt))
                 + JITTER * torch.eye(N, dtype=DTYPE))
            self.L = torch.linalg.cholesky(K)
            self.alpha = torch.cholesky_solve(self.yt, self.L)

    def predict(self, x_raw):
        """Return (mean, epistemic_var, aleatoric_var) on the ORIGINAL y scale."""
        with torch.no_grad():
            Xn = torch.tensor(((np.asarray(x_raw).flatten() - self.X_min) / self.rx).reshape(-1, 1),
                              dtype=DTYPE)
            Ks = self.model.kernel(self.Xt, Xn)
            Kss = self.model.kernel(Xn, Xn)
            v = torch.cholesky_solve(Ks, self.L)
            mu = (Ks.T @ self.alpha).flatten()
            epi = torch.diag(Kss - Ks.T @ v).clamp_min(0)
            ale = self.model.sigma2(Xn)
        return (mu.numpy() * self.ry + self.y_min, epi.numpy() * self.ry ** 2, ale.numpy() * self.ry ** 2)


class PerCategoryHGPR:
    """5 independent HGPRs. cold_fit() once before the BO loop, then refit_level() (warm) the one
    category that got a new point each iteration. Same predict/mean_std/r interface as PerCategoryGPs."""

    def __init__(self, levels):
        self.levels = sorted(int(lv) for lv in levels)
        self._lv = {}

    @classmethod
    def cold_fit(cls, data):
        o = cls(list(data))
        for lv, d in data.items():
            h = _LevelHGPR()
            x, y = _flatten(d)
            h.fit(x, y, EPOCHS_COLD)
            o._lv[int(lv)] = h
        return o

    def refit_level(self, lv, d):
        """Warm-refit ONLY this category (keeps its learned params as the init)."""
        x, y = _flatten(d)
        self._lv[int(lv)].fit(x, y, EPOCHS_WARM)

    def predict(self, lv, xs, observation_noise=False):
        mu, epi, ale = self._lv[int(lv)].predict(xs)
        return mu, (epi + ale if observation_noise else epi)

    def mean_std(self, lv, xs):
        mu, epi, _ = self._lv[int(lv)].predict(xs)
        return mu, np.sqrt(epi)

    def r(self, lv, xs):
        _, _, ale = self._lv[int(lv)].predict(xs)
        return ale
