"""
acquisitions.py -- the 6 acquisition families, ported from Heter_BO_GF/acquisition_func.m.
(Verbatim from study_v2_gp/utils/acquisitions.py -- problem-independent.)

MINIMIZATION problem: every function returns the MATLAB `U_negate` that the optimizer
MINIMIZES (lower = more desirable). bo.py uses ONE GLOBAL incumbent ymin (best observed
sample-mean across ALL categories) so values are comparable across categories.

Inputs (broadcastable numpy arrays): mu (posterior mean), s (epistemic std, clamped),
r (predicted aleatoric VARIANCE, hetero acqs only), ymin (global incumbent).
knobs: haei->gamma, anpei->beta_anpei, rahbo->alpha (beta fixed = 2, as in study_driver.m).
"""
import numpy as np
from scipy.stats import norm

BETA_RAHBO = 2.0


def _ei(mu, s, ymin):
    s = np.maximum(s, 1e-12)
    b = (ymin - mu) / s
    return (ymin - mu) * norm.cdf(b) + s * norm.pdf(b)


def ei(mu, s, ymin):
    return -_ei(mu, s, ymin)


def lcb(mu, s):
    return mu - 2.0 * s


def pi(mu, s, ymin):
    s = np.maximum(s, 1e-12)
    b_pi = (ymin - mu - 0.01) / s
    return -norm.cdf(b_pi)


def haei(mu, s, r, ymin, gamma):
    EI = _ei(mu, s, ymin)
    var_epi = np.maximum(s, 1e-12) ** 2
    scale = 1.0 - (gamma * np.sqrt(r)) / np.sqrt(var_epi + gamma ** 2 * r)
    scale = np.maximum(scale, 0.0)
    return -(EI * scale)


def anpei(mu, s, r, ymin, beta_anpei):
    EI = _ei(mu, s, ymin)
    ale_std = np.sqrt(np.maximum(r, 1e-12))
    return -(beta_anpei * EI - (1.0 - beta_anpei) * ale_std)


def rahbo(mu, s, r, ymin, alpha, beta=BETA_RAHBO):
    lcb_f = mu - beta * s
    return lcb_f + alpha * r


_NEEDS_R = {"haei", "anpei", "rahbo"}


def needs_aleatoric(acf):
    return acf in _NEEDS_R


def evaluate(acf, mu, s, ymin, r=None, param=None):
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


# ---- acquisition-config metadata (problem-independent) ----
CONFIG_ORDER = [
    ("lcb", float("nan")), ("pi", float("nan")), ("ei", float("nan")),
    ("haei", 0.5), ("haei", 1.0), ("haei", 5.0),
    ("anpei", 0.2), ("anpei", 0.5), ("anpei", 0.8),
    ("rahbo", 0.5), ("rahbo", 1.0), ("rahbo", 5.0),
]
NOISE_BLIND = [("lcb", float("nan")), ("pi", float("nan")), ("ei", float("nan"))]
_KNOB = {"haei": "g", "rahbo": "a", "anpei": "b"}


def label(acf, param):
    if acf == "haei":  return f"HAEI(γ={param:g})"
    if acf == "rahbo": return f"RAHBO(α={param:g})"
    if acf == "anpei": return f"ANPEI(β={param:g})"
    return acf.upper()


def acf_tag(acf, param):
    if param != param:
        return acf
    return f"{acf}_{_KNOB.get(acf, 'p')}{param:g}"


def canon_cfg(acf, param):
    return (acf, "na" if param != param else round(float(param), 6))
