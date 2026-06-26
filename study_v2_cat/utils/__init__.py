"""
utils -- joint categorical-GP Bayesian-optimization study (study_v2_cat, "Method C").

ONE MixedSingleTaskGP + CategoricalKernel models all categories jointly (shares across
levels, like LVGP but in botorch). Plotting reuses study_v2's StudyResults gallery (the
saved .npz schema is identical), plus compare_studies()/compare_studies_multi() overlays:

    from utils import StudyResults, compare_studies_multi, problem
    cat  = StudyResults.load("results")                    # this study (categorical GP)
    gp   = StudyResults.load("../study_v2_gp/results")     # per-category GP
    lvgp = StudyResults.load("../study_v2/results")        # LVGP (.mat)
    compare_studies_multi([(lvgp,"LVGP"), (gp,"Per-category GP"), (cat,"Categorical GP")],
                          metric="true_regret", n_rep=10)  # the 3-way headline

`problem` is light (numpy/scipy). The BO worker path imports `utils.bo` directly, which
does NOT trigger the results/plotting stack -- workers stay lean.
"""
from . import problem

__all__ = ["StudyResults", "compare_studies", "compare_studies_multi",
           "compare_summary_heatmaps", "problem"]


def __getattr__(name):                            # PEP 562: lazy import of the plotting stack
    if name in ("StudyResults", "compare_studies", "compare_studies_multi",
                "compare_summary_heatmaps"):
        import importlib
        return getattr(importlib.import_module(".results", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
