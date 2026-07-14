# LaTeX comparison tables

Six tables of the full **acquisition × surrogate-model** grid, one per (metric × n_rep), matching the
`acq_method_tables.xlsx` sheets. Generated from `utils/results.py::export_latex_tables` (same data as the
notebook / Excel).

- `regret_nrep03.tex`, `regret_nrep10.tex` — final regret (value − f*), lower is better
- `noise_nrep03.tex`,  `noise_nrep10.tex`  — final σ² at the incumbent, lower is better
- `mv_nrep03.tex`,     `mv_nrep10.tex`     — mean-variance regret (RAHBO robustness, MV=f+ασ²), lower is better
- `main.tex` — preamble + `\input`s all four; compile it for `main.pdf`

**Highlighting** (per problem column): **bold** = best model for that acquisition; **green** = the single
best (acquisition, model) combination.

Regenerate + compile:
```python
from utils.results import GridResults, export_latex_tables
export_latex_tables(GridResults.load('results'), out_dir='latex', n_reps=(3,10), compile_pdf=True)
```
Or by hand: `cd latex && pdflatex main.tex`. Needs only `booktabs`, `xcolor`, `amsmath`, `geometry`
(no `multirow`/`colortbl`). Each `*.tex` is a standalone `table` float you can `\input` into a paper.
