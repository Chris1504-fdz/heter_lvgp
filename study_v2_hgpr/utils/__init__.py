"""
utils -- per-category GP Bayesian-optimization study (study_v2_gp, "Part 2").

Plotting reuses study_v2's full StudyResults gallery (the saved .mat schema is identical),
plus a compare_studies() overlay for the head-to-head LVGP-vs-GP figures:

    from utils import StudyResults, compare_studies, problem
    gp   = StudyResults.load("results")                       # this study (per-category GP)
    lvgp = StudyResults.load("../study_v2/results")           # the LVGP study
    gp.plot_convergence_true(log=True)                        # any of the ~20 standard plots
    compare_studies(gp, lvgp, metric="true_regret", n_rep=10) # the Part-2 headline

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
