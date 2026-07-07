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


def get(name) -> ModelInfo:
    if name not in MODELS:
        raise KeyError(f"unknown model '{name}'. available: {list(MODELS)}")
    return MODELS[name]
