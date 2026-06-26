"""
utils -- joint categorical-GP BO study driven by BoTorch optimize_acqf_mixed (study_v2_acqf).

Same MixedSingleTaskGP + CategoricalKernel as study_v2_cat (one model, shares across levels),
but the acquisition is optimized with BoTorch's optimize_acqf_mixed (model fit on -y) instead
of the hand-rolled grid -- see utils/acqf.py and utils/bo.py. The saved .npz schema is identical,
so study_v2's StudyResults gallery + compare_studies()/compare_studies_multi() overlays load it:

    from utils import StudyResults, compare_studies_multi, problem
    acqf = StudyResults.load("results")                    # this study (botorch optimize_acqf_mixed)
    cat  = StudyResults.load("../study_v2_cat/results")    # joint cat GP, hand-rolled optimizer
    lvgp = StudyResults.load("../study_v2/results")        # LVGP (.mat)
    compare_studies_multi([(lvgp,"LVGP"), (cat,"Categorical GP"), (acqf,"BoTorch acqf")],
                          metric="true_regret", n_rep=10)

`problem` is light (numpy/scipy). The BO worker path imports `utils.bo` directly, which
does NOT trigger the results/plotting stack -- workers stay lean.
"""
from . import problem

__all__ = ["StudyResults", "compare_studies", "compare_studies_multi",
           "compare_summary_heatmaps", "compare_variance_convergence",
           "compare_runtime", "runtime_summary", "problem"]


def __getattr__(name):                            # PEP 562: lazy import of the plotting stack
    if name in ("StudyResults", "compare_studies", "compare_studies_multi",
                "compare_summary_heatmaps", "compare_variance_convergence",
                "compare_runtime", "runtime_summary"):
        import importlib
        return getattr(importlib.import_module(".results", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
