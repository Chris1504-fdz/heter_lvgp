"""
acqf.py -- BoTorch acquisition factory for study_v2_acqf.

The point of this study: optimize the acquisition with BoTorch's `optimize_acqf_mixed`
instead of the hand-rolled grid + L-BFGS used in the other studies. Acquisitions:
  ei  -> botorch LogExpectedImprovement
  lcb -> botorch UpperConfidenceBound (beta=4 -> mean + sqrt(4)*sigma = mean + 2*sigma;
         on the -y model this is -mu_f + 2*sigma = minimizing mu_f - 2*sigma, i.e. LCB)
  pi  -> botorch ProbabilityOfImprovement
  haei / anpei / rahbo -> custom AnalyticAcquisitionFunction subclasses (NOT in botorch),
         which combine the GP posterior with the aleatoric r(x), ported from acquisition_func.m.

Everything is in BoTorch's MAXIMIZE convention; the model is fit on -y (see model.py), so
mean=mu_model=-mu_f and best_f = max(-y_obs) = -min(y_obs). The custom EI used by haei/anpei
is the plain analytic EI (matching the MATLAB code, not LogEI).
"""
import numpy as np
import torch
from torch.distributions import Normal

from botorch.acquisition.analytic import (LogExpectedImprovement, UpperConfidenceBound,
                                          ProbabilityOfImprovement, AnalyticAcquisitionFunction)
from botorch.utils.transforms import t_batch_mode_transform

DTYPE = torch.float64
BETA_RAHBO = 2.0
_STD_NORMAL = Normal(torch.tensor(0.0, dtype=DTYPE), torch.tensor(1.0, dtype=DTYPE))


class TorchAleatoric:
    """Differentiable r(x1, code) wrapping the per-category numpy aleatoric polynomials
    (utils.aleatoric.AleatoricModels). code = level-1 (0-based)."""

    def __init__(self, ale, levels):
        levels = sorted(int(lv) for lv in levels)
        self.degree = ale.models[levels[0]].degree
        self.mu = torch.tensor([ale.models[lv].mu for lv in levels], dtype=DTYPE)
        self.sd = torch.tensor([ale.models[lv].sd for lv in levels], dtype=DTYPE)
        self.theta = torch.tensor(np.stack([ale.models[lv].theta for lv in levels]), dtype=DTYPE)

    def r(self, x1, code):
        code = code.clamp(0, self.mu.shape[0] - 1)
        mu = self.mu[code]; sd = self.sd[code]; th = self.theta[code]    # (b,), (b,), (b, deg+1)
        wn = (x1 - mu) / sd
        phi = torch.stack([wn ** d for d in range(self.degree + 1)], dim=-1)
        return torch.clamp(torch.exp(2 * (th * phi).sum(-1)), min=1e-12)  # variance r(x)


def _mu_s(model, X):
    """Posterior mean (= -mu_f) and epistemic std at X (shape (b,1,d)) -> each (b,)."""
    post = model.posterior(X)
    mu = post.mean.squeeze(-1).squeeze(-1)
    s = post.variance.clamp_min(1e-12).sqrt().squeeze(-1).squeeze(-1)
    return mu, s


def _ei(mu, s, best_f):
    """Plain analytic EI in maximize convention (matches acquisition_func.m's EI)."""
    z = (mu - best_f) / s
    return (mu - best_f) * _STD_NORMAL.cdf(z) + s * torch.exp(_STD_NORMAL.log_prob(z))


class HAEI(AnalyticAcquisitionFunction):
    """EI * (1 - gamma*sqrt(r)/sqrt(s^2 + gamma^2 r))  (Griffiths et al., het-aware EI)."""
    def __init__(self, model, best_f, gamma, ale):
        super().__init__(model=model)
        self.best_f = float(best_f); self.gamma = float(gamma); self.ale = ale

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X):
        mu, s = _mu_s(self.model, X)
        ei = _ei(mu, s, self.best_f)
        r = self.ale.r(X[..., 0, 0], X[..., 0, 1].round().long())
        scale = (1 - self.gamma * r.sqrt() / torch.sqrt(s ** 2 + self.gamma ** 2 * r)).clamp_min(0.0)
        return ei * scale


class ANPEI(AnalyticAcquisitionFunction):
    """beta*EI - (1-beta)*sqrt(r)  (Griffiths et al.)."""
    def __init__(self, model, best_f, beta_anpei, ale):
        super().__init__(model=model)
        self.best_f = float(best_f); self.beta = float(beta_anpei); self.ale = ale

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X):
        mu, s = _mu_s(self.model, X)
        ei = _ei(mu, s, self.best_f)
        r = self.ale.r(X[..., 0, 0], X[..., 0, 1].round().long())
        return self.beta * ei - (1 - self.beta) * r.sqrt()


class RAHBO(AnalyticAcquisitionFunction):
    """Maximize mu_model + beta*s - alpha*r  ==  minimize (mu_f - beta*s) + alpha*r
    (the minimization analogue of RAHBO; mu_model = -mu_f on the -y model)."""
    def __init__(self, model, alpha, ale, beta=BETA_RAHBO):
        super().__init__(model=model)
        self.alpha = float(alpha); self.beta = float(beta); self.ale = ale

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X):
        mu, s = _mu_s(self.model, X)
        r = self.ale.r(X[..., 0, 0], X[..., 0, 1].round().long())
        return mu + self.beta * s - self.alpha * r


def make_acqf(acf, param, model, best_f, ale):
    """Return a BoTorch acquisition (maximize convention) for `acf`. `ale` is a TorchAleatoric
    (only used by haei/anpei/rahbo). best_f = max(-y_observed)."""
    acf = acf.lower()
    if acf == "ei":    return LogExpectedImprovement(model, best_f=best_f)
    if acf == "lcb":   return UpperConfidenceBound(model, beta=4.0)        # sqrt(4)=2 -> mean + 2*sigma
    if acf == "pi":    return ProbabilityOfImprovement(model, best_f=best_f)
    if acf == "haei":  return HAEI(model, best_f, param, ale)
    if acf == "anpei": return ANPEI(model, best_f, param, ale)
    if acf == "rahbo": return RAHBO(model, param, ale)
    raise ValueError(f"unknown acf {acf!r}")
