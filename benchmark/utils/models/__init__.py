"""
Model registry for the 9x4 grid. Each entry declares its ENGINE (python | matlab), which
acquisitions it supports, and -- for python models -- the class implementing the interface.
Adding a python model = one file + one line here.
"""
from dataclasses import dataclass, field

from .separate_gp import SeparateGP
from .categorical_kernel import CategoricalKernelGP

FULL = ("ei", "lcb", "pi", "haei", "anpei", "rahbo")   # noise-aware models
BLIND = ("ei", "lcb", "pi")                             # noise-unaware (standard LVGP)


@dataclass
class ModelInfo:
    name: str
    engine: str                 # "python" | "matlab"
    supports: tuple             # acquisition families this model runs
    cls: object = None          # python model class implementing the interface (None for matlab)
    matlab_name: str = ""       # id passed to matlab/study_driver.m (matlab models only)
    label: str = ""             # display label for plots
    aux: bool = False           # auxiliary model (LVGP-validation only): excluded from the default
                                # benchmark grid + progress counts; still runnable via --models


MODELS = {
    "separate_gp":        ModelInfo("separate_gp", "python", FULL, cls=SeparateGP,
                                    label="Separate GP"),
    "categorical_kernel": ModelInfo("categorical_kernel", "python", FULL, cls=CategoricalKernelGP,
                                    label="Categorical kernel"),
    "standard_LVGP":      ModelInfo("standard_LVGP", "matlab", BLIND, matlab_name="standard_lvgp",
                                    label="Standard LVGP"),
    "heter_LVGP":         ModelInfo("heter_LVGP", "matlab", FULL, matlab_name="heter_lvgp",
                                    label="Heteroscedastic LVGP"),
}


# Python LVGP analogues of the MATLAB engines. The NATIVE port faithfully reproduces MATLAB's
# LVGP_fit(_noise) treatment (min-max norm, noise-on-correlation, eps-ladder eigenvalue nugget,
# profiled sigma2) -- it matches MATLAB and does not get stuck, unlike the lvgp-bayes wrapper whose
# FixedNoiseGaussianLikelihood + standardization + horseshoe priors diverged. Native is the default.
# aux=True: these are for the LVGP-vs-MATLAB validation study only. They are kept OUT of the default
# benchmark grid (run.py) and the progress/leaderboard counts, so the 4-model completed problems do
# not read as "half done". Run them explicitly with e.g. `--models lvgp_native heter_lvgp_native`.
from .hetero_lvgp_native import LVGPNative, HeterLVGPNative
MODELS["lvgp_native"] = ModelInfo("lvgp_native", "python", BLIND, cls=LVGPNative,
                                  label="LVGP (native)", aux=True)
MODELS["heter_lvgp_native"] = ModelInfo("heter_lvgp_native", "python", FULL, cls=HeterLVGPNative,
                                        label="Hetero LVGP (native)", aux=True)

# lvgp-bayes wrapper kept available (guarded) for A/B reference, but NOT the default python LVGP.
try:
    from .lvgp_torch import LVGPTorch, HeterLVGPTorch
    MODELS["lvgp_torch"] = ModelInfo("lvgp_torch", "python", BLIND, cls=LVGPTorch,
                                     label="LVGP (torch)", aux=True)
    MODELS["heter_lvgp_torch"] = ModelInfo("heter_lvgp_torch", "python", FULL,
                                           cls=HeterLVGPTorch, label="Hetero LVGP (torch)", aux=True)
except ImportError:
    pass


# The core benchmark grid: the 4 models the sweep and all progress/leaderboard views count by default.
BENCHMARK_MODELS = [m for m, mi in MODELS.items() if not mi.aux]


def get(name) -> ModelInfo:
    if name not in MODELS:
        raise KeyError(f"unknown model '{name}'. available: {list(MODELS)}")
    return MODELS[name]
