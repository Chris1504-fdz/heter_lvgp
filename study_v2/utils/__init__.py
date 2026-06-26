"""
utils — plotting package for the heteroscedastic LVGP BO study (study_v2).

    from utils import StudyResults
    study = StudyResults.load("results")
    study.summary()
    study.plot_convergence()        # every plot_* method returns a Figure
"""
from .results import StudyResults
from . import problem

__all__ = ["StudyResults", "problem"]
