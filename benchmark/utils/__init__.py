"""
benchmark utils -- shared code for the 9-function x 4-model mixed-variable BO grid.

  from utils import problems as P, acquisitions, bo
  from utils.models import MODELS, get as get_model

`problems` and `acquisitions` are light (numpy/scipy). The python models (utils.models) pull in
torch/botorch; the BO worker imports what it needs. results.py (loading/plotting) is imported lazily.
"""
from . import problems, acquisitions
