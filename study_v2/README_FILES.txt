STUDY FOLDER — WHAT'S WHAT
==========================

RUN THE STUDY (MATLAB — must stay in this folder; they use ../LVGP_Matlab_codes
and ../Heter_BO_GF via relative paths):
  study_driver.m        runs ONE BO cell (acq, n_rep, seed) and saves its .mat
  run_all.m             sequential: runs every pending cell (resumable, 1 license)
  run_all_parallel.m    parallel: same, across a local parpool (faster; needs
                        Parallel Computing Toolbox)

VISUALIZE THE FIGURES — interactive (recommended):
  analysis.ipynb        notebook to view the main plots inline. Select the
                        ml_gp_env kernel, Run All. Run it from THIS folder.
  utils/plot_utils.py   all the plotting logic (each function returns a Figure);
                        the notebook just imports this and calls it.

MAKE THE FIGURES — batch scripts (write PNGs to plots/, run with ml_gp_env):
  analyze.py            the 4 core plots  -> plots/main/
  suggest_plots.py      the 7-plot analysis gallery -> plots/suggestions/
  plot_single_run.py    per-category "slice" view of one run -> plots/single_runs/
  plot_best_designs.py  best-design-per-seed view (plot 2 variant) -> plots/main/
  run_sweep.py          (utility) rebuilds sweep_results.csv: run_sweep.py --collect-only

PROGRESS TRACKING (optional — TensorBoard):
  tb_log.py             writes scalars + plot images to tb/  (one-shot)
  tb_loop.sh            re-runs tb_log periodically
  tb/                   TensorBoard event files (~240 MB — delete if not using TB)
  .tb_conditions.json   manifest of what's been logged

DATA & OUTPUTS:
  results/              all run results: <acq>/nrep<NN>/seed<NN>.mat  (the data)
  plots/                generated figures (main/ suggestions/ single_runs/ diagnostics/)
  sweep_results.csv     tidy long-format table of every run's convergence

PRESENTATION (your slides):
  presentation.pptx
  make_slides.py

_review_trash/          old logs + obsolete scripts moved aside; see its README,
                        then delete.

NOTE: do NOT move the .m driver files into subfolders — they locate the model
code via relative paths and writing them elsewhere breaks that.
