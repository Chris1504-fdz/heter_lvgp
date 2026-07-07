# benchmark/ — 9-function × 4-model mixed-variable BO grid

A de-duplicated, registry-driven comparison of **4 surrogate models** across **9 mixed-variable
(1-D continuous × K-level categorical) test functions**, replacing the copy-per-study pattern of
the `study_v2*` directories. Code is shared in `utils/` + `matlab/`; `results/<function>/<model>/…`
holds only data.

## Models (the `--models` axis)
| model | engine | acquisitions | code |
|---|---|---|---|
| `separate_gp` | python | ei/lcb/pi/haei/anpei/rahbo | `utils/models/separate_gp.py` |
| `categorical_kernel` | python | full | `utils/models/categorical_kernel.py` |
| `standard_LVGP` | **matlab** | ei/lcb/pi (noise-blind) | `matlab/standard_driver.m` |
| `heter_LVGP` | **matlab** | full | `matlab/heter_driver.m` |

Two engines, one schema: python models → `.npz`, matlab (LVGP) models → `.mat`; `utils/results.py`
reads both. Adding a python model = one file + one line in `utils/models/__init__.py`.

## Functions (the `--functions` axis)
Defined in `utils/problems.py` (a `ProblemSpec` per function: `f(x1,level)`, `sigma(x1,level)`,
domain, `n_levels`). `branin_hetero` is the worked reference; **8 stubs await equations**. Each
function MUST be mirrored in `matlab/problems.m` for the two LVGP models — `verify_problems.py`
checks the Python and MATLAB definitions agree.

> **To add a function:** fill a `ProblemSpec` in `utils/problems.py` AND a `case` in
> `matlab/problems.m` (reshape any level-indexed lookup to `size(x1)` — see the Branin comment),
> then run `python verify_problems.py`.

## Run
```bash
PY=/data/zhq7531/envs/ml_gp_env/bin/python
cd /data/zhq7531/IDEAL/hetero_lvgp/benchmark

$PY run.py --toy                                   # small probe: branin × 4 models × ei × seeds 1-2
$PY run.py --seeds 30 --workers 8                  # full grid over all DEFINED functions × 4 models
$PY run.py --functions branin_hetero --models separate_gp categorical_kernel --seeds 30
$PY run.py --collect-only                          # -> sweep_results.csv
# then open notebooks/compare.ipynb (ml_gp_env kernel)
```
`run.py` dispatches by engine: a `ProcessPoolExecutor` runs python cells; MATLAB launches are
throttled (≥8 s apart) + timed out so a burst can't wedge the MathWorks Service Host (see
`hetero_lvgp/study_v2_plain_lvgp/README.md` for the MSH-reset recipe if it ever hangs). Resumable —
cells whose `.npz`/`.mat` exist are skipped.

## Layout
```
utils/ problems.py · acquisitions.py · bo.py · results.py · models/{base,separate_gp,categorical_kernel,__init__}.py
matlab/ problems.m · heter_driver.m · standard_driver.m · heter_lvgp/ · standard_lvgp/
run.py · verify_problems.py · notebooks/compare.ipynb · results/<function>/<model>/<acq>/nrep<NN>/seed<NN>.{npz,mat}
```
Headline metric: `true_best_sampled` (min f_true over sampled points). The six `study_v2*`
directories are left untouched as reference.
