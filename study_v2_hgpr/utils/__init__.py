"""
utils -- per-category heteroscedastic GP (HGPR) Bayesian-optimization study (study_v2_hgpr).

Surrogate = Method A of heterockedastic_new_inv/heterosk.ipynb (Ozbayram et al., CMAME 2024):
one HGPR per category, each learning its polynomial noise model sigma^2(x) jointly via a MAP
loss (see utils/model.py and utils/bo.py). Saved .npz schema is identical to the other studies,
so study_v2's StudyResults gallery loads it and the top-level comparison overlays include it:

    from utils import StudyResults, compare_studies, problem
    hgpr = StudyResults.load("results")                       # this study (per-category HGPR)
    gp   = StudyResults.load("../study_v2_gp/results")        # per-category GP + separate noise poly
    compare_studies(hgpr, gp, metric="true_regret", n_rep=10)

`problem` is light (numpy/scipy). The BO worker path imports `utils.bo` directly, which
does NOT trigger the results/plotting stack -- workers stay lean.
"""
from . import problem

__all__ = ["StudyResults", "compare_studies", "problem"]


def __getattr__(name):                            # PEP 562: lazy import of the plotting stack
    if name == "StudyResults":
        from .results import StudyResults
        return StudyResults
    if name == "compare_studies":
        from .results import compare_studies
        return compare_studies
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
