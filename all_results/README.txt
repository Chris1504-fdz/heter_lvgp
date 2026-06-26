all_results/ — raw per-run result files from the three BO studies (copies).

Layout:  all_results/<study>/<acquisition>/nrep<NN>/seed<NN>.<ext>
  study_v2/      LVGP (MATLAB)         -> .mat
  study_v2_gp/   per-category GP       -> .npz
  study_v2_cat/  categorical GP (Method C) -> .npz

Every file holds the full BO trajectory with the SAME field names across studies:
  X_sampled [x1, level], Y_sampled, Y_var_sampled, Y_min_history, X_min_est,
  X_best_final, n_initial, var_fctr, meta{acf,acf_param,n_rep,seed,num_iter,runtime,model}.
Load: scipy.io.loadmat (.mat) | numpy.load(..., allow_pickle=True) (.npz),
or via study_v2_cat/utils/results.py :: StudyResults.load("all_results/<study>").
