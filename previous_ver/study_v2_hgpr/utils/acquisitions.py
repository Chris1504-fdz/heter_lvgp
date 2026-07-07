"""
acquisitions.py -- the 6 acquisition families, ported from Heter_BO_GF/acquisition_func.m.

MINIMIZATION problem (smaller objective is better), so the incumbent improvement is
(y_min - mu). Every function returns the MATLAB `U_negate`: the scalar that fmincon
MINIMIZES, i.e. LOWER = MORE DESIRABLE. bo.py minimizes it and -- crucially -- uses ONE
GLOBAL incumbent y_min (best observed sample-mean across ALL categories) for every
category's acquisition, so the values are comparable across categories.

Inputs (all broadcastable numpy arrays):
    mu   : per-category GP posterior mean
    s    : per-category GP epistemic std (clamped >= 1e-12)
    r    : predicted aleatoric VARIANCE r(x) from aleatoric.py (hetero acqs only)
    ymin : global incumbent = min observed sample-mean
knobs: haei->gamma, anpei->beta_anpei, rahbo->alpha (beta fixed = 2, as in study_driver.m).
"""
import numpy as np
from scipy.stats import norm

BETA_RAHBO = 2.0       # bo_options.beta in study_driver.m (fixed for the rahbo lcb_f term)


def _ei(mu, s, ymin):
    """Expected improvement for minimization: (ymin-mu)Phi(b) + s phi(b), b=(ymin-mu)/s."""
    s = np.maximum(s, 1e-12)
    b = (ymin - mu) / s
    return (ymin - mu) * norm.cdf(b) + s * norm.pdf(b)


# ---- each returns U_negate (minimize; lower = better) ----
def ei(mu, s, ymin):
    return -_ei(mu, s, ymin)


def lcb(mu, s):
    return mu - 2.0 * s                                       # lower confidence bound (minimized)


def pi(mu, s, ymin):
    s = np.maximum(s, 1e-12)
    b_pi = (ymin - mu - 0.01) / s                            # 0.01 margin, as in MATLAB
    return -norm.cdf(b_pi)


def haei(mu, s, r, ymin, gamma):
    """Heteroscedastic augmented EI (Griffiths et al.): EI * (1 - gamma*sqrt(r)/sqrt(s^2+gamma^2 r))."""
    EI = _ei(mu, s, ymin)
    var_epi = np.maximum(s, 1e-12) ** 2
    scale = 1.0 - (gamma * np.sqrt(r)) / np.sqrt(var_epi + gamma**2 * r)
    scale = np.maximum(scale, 0.0)                           # HAEI is non-negative
    return -(EI * scale)


def anpei(mu, s, r, ymin, beta_anpei):
    """ANPEI: beta*EI - (1-beta)*sqrt(r)."""
    EI = _ei(mu, s, ymin)
    ale_std = np.sqrt(np.maximum(r, 1e-12))
    return -(beta_anpei * EI - (1.0 - beta_anpei) * ale_std)


def rahbo(mu, s, r, ymin, alpha, beta=BETA_RAHBO):
    """Risk-averse heteroscedastic BO (minimization analogue): (mu - beta*s) + alpha*r."""
    lcb_f = mu - beta * s                                    # optimistic for minimization
    return lcb_f + alpha * r                                 # lcb_var ~= r (point aleatoric estimate)


# ---- dispatcher: (acf, knob) -> U_negate ----
_NEEDS_R = {"haei", "anpei", "rahbo"}


def needs_aleatoric(acf):
    return acf in _NEEDS_R


def evaluate(acf, mu, s, ymin, r=None, param=None):
    """U_negate for acquisition `acf`. `param` is the family knob (gamma/beta_anpei/alpha);
    ignored for ei/lcb/pi. `r` (aleatoric variance) is required for haei/anpei/rahbo."""
    acf = acf.lower()
    if acf == "ei":    return ei(mu, s, ymin)
    if acf == "lcb":   return lcb(mu, s)
    if acf == "pi":    return pi(mu, s, ymin)
    if acf in _NEEDS_R and r is None:
        raise ValueError(f"acquisition '{acf}' requires aleatoric variance r(x)")
    if acf == "haei":  return haei(mu, s, r, ymin, gamma=param)
    if acf == "anpei": return anpei(mu, s, r, ymin, beta_anpei=param)
    if acf == "rahbo": return rahbo(mu, s, r, ymin, alpha=param)
    raise ValueError(f"Unknown acquisition function: {acf}")
