function heter_driver(func_name, acf, acf_param, n_rep, seed, num_iter, out_file, doe_file)
% Heteroscedastic-LVGP engine for the benchmark grid (generalizes study_v2/study_driver.m).
% Builds the problem from problems.m (any function name), runs the Heter_BO_GF BO, saves the
% study_driver.m field schema to out_file. `level` is the categorical (1..n_lv); var_fctr is the
% IDENTITY 1:n_lv so the objective handle receives [x1, level] and problems.m does the mapping.
%   acf : ei|lcb|pi|haei|anpei|rahbo   acf_param: knob (NaN for ei/lcb/pi)

    here = fileparts(mfilename('fullpath'));
    addpath(here);                                   % problems.m
    addpath(fullfile(here, 'heter_lvgp'));           % Heter_BO_GF + LVGP_Matlab_codes
    try, pset = parallel.Settings; pset.Pool.AutoCreate = false; catch, end
    rng(seed);

    [f, sig, lb, ub, n_lv] = problems(func_name);
    ind_qual = 2; X_range_continuous = [lb; ub];   % DOE (incl. its size) comes from doe_file
    var_fctr = 1:n_lv;                               % identity -> objfunc gets [x1, level]
    obj_noisy = @(X) f(X(:,1), X(:,2)) + randn(size(X,1),1) .* sig(X(:,1), X(:,2));

    % ---- initial DOE: SHARED design generated in Python (utils/doe_cache.py) -- common random numbers,
    %      byte-identical for every model. Loads X_sample, Y_sample, Var_sample, Y_rep. ----
    D = load(doe_file);
    X_sample = D.X_sample; Y_sample = D.Y_sample(:); Var_sample = D.Var_sample(:);
    n_tr = size(X_sample, 1); Y_rep_sample = cell(n_tr, 1);
    for row = 1:n_tr
        Y_rep_sample{row} = D.Y_rep(row, :)';       % n_rep noisy replicates for this design point
    end

    custom_points = 0; model_options.ind_qual = ind_qual; model_options.dim_z = 2;
    bo_options.n_rep = n_rep; bo_options.poly_degree = 2; bo_options.poly_lambda = 1e-3;
    bo_options.PlotOrNot = 'DontPlot';
    switch lower(acf)
        case 'haei',  bo_options.gamma = acf_param;
        case 'rahbo', bo_options.alpha = acf_param; bo_options.beta = 2;
        case 'anpei', bo_options.beta_anpei = acf_param;
        case {'ei','lcb','pi'}
        otherwise, error('Unknown acf: %s', acf);
    end

    t0 = tic;
    result = bayesian_optimizer(obj_noisy, var_fctr, X_sample, Y_sample, Var_sample, ...
        Y_rep_sample, X_range_continuous, acf, num_iter, custom_points, model_options, bo_options);
    runtime = toc(t0);

    Y_min_history      = result.Y_min_history(:)';
    X_sampled          = result.X_sampled;
    Y_sampled          = result.Y_sampled(:)';
    Y_var_sampled      = result.Y_var_sampled(:)';
    X_next_history     = result.X_next_history;
    Y_next_history     = result.Y_next_history(:)';
    Y_var_next_history = result.Y_var_next_history(:)';
    Y_min_est          = result.Y_min_est(:)';
    X_min_est          = result.X_min_est;
    X_best_final       = result.X_best_final;
    Y_best_final       = result.Y_best_final;
    Y_var_best_final   = result.Y_var_best_final;
    n_initial          = size(X_sample, 1);

    % ---- SCHEMA v2: everything post-processing needs, so a cell never has to be re-run ----
    acf_val   = result.acf_val(:)';                    % acquisition value at the chosen point
    mu_at_est = result.Mu_at_est(:)';                  % posterior mean      @ X_min_est
    s_at_est  = result.S_at_est(:)';                   % epistemic std       @ X_min_est
    r_at_est  = result.R_at_est(:)';                   % aleatoric variance  @ X_min_est

    Y_rep_sampled = cell2mat(cellfun(@(v) v(:)', result.Y_rep_sampled, 'UniformOutput', false));
    X_init = X_sample; Y_init = Y_sample(:)'; Y_rep_init = D.Y_rep;

    % noise-free objective + true noise std at every sampled point: keeps the cell re-scorable even
    % if problems.m is later edited.
    f_true_sampled     = reshape(f(X_sampled(:,1),   X_sampled(:,2)), 1, []);
    sigma_true_sampled = reshape(sig(X_sampled(:,1), X_sampled(:,2)), 1, []);

    % Curated fitted hyperparameters -- deliberately NOT result.final_model, whose fit_detail embeds
    % R and Linv (n_tr x n_tr => ~3.6 GB over the Phase-2 grid) and which are recomputable from these.
    fm = result.final_model;
    hyper = struct();
    hyper.z           = fm.qual_param.z;               % latent embedding of the categorical levels
    hyper.phi         = fm.quant_param.phi;            % continuous lengthscales
    hyper.beta_hat    = fm.fit_detail.beta_hat;
    hyper.sigma2      = fm.fit_detail.sigma2;
    hyper.nug_opt     = fm.fit_detail.nug_opt;
    hyper.poly_theta  = fm.aleatoric_poly.theta;       % heteroscedastic noise polynomial
    hyper.poly_degree = fm.aleatoric_poly.degree;
    hyper.poly_lambda = fm.aleatoric_poly.lambda;
    hyper.poly_Z_latent = fm.aleatoric_poly.Z_latent;

    meta = struct('problem',func_name,'model','heter_LVGP','acf',acf,'acf_param',acf_param, ...
                  'n_rep',n_rep,'seed',seed,'num_iter',num_iter,'runtime',runtime, ...
                  'schema_version',2,'doe_mode','slhd','n_init',n_initial,'n_levels',n_lv, ...
                  'lb',lb,'ub',ub,'timestamp',datestr(now,'yyyy-mm-ddTHH:MM:SS'), ...
                  'noise_stream','matlab:rng(seed)');

    % ATOMIC write: save to a temp file in the destination dir, then rename over the target. A kill
    % or timeout mid-save can otherwise leave a truncated .mat that run.py skips forever.
    [out_dir, out_base] = fileparts(out_file);
    tmp_file = fullfile(out_dir, sprintf('%s.tmp%d.mat', out_base, feature('getpid')));
    save(tmp_file, 'Y_min_history','X_sampled','Y_sampled','Y_var_sampled', ...
        'X_next_history','Y_next_history','Y_var_next_history','Y_min_est','X_min_est', ...
        'X_best_final','Y_best_final','Y_var_best_final','n_initial','var_fctr','meta', ...
        'Y_rep_sampled','acf_val','mu_at_est','s_at_est','r_at_est', ...
        'X_init','Y_init','Y_rep_init','f_true_sampled','sigma_true_sampled','hyper','-v7');
    movefile(tmp_file, out_file, 'f');
    fprintf('DONE %s heter_LVGP acf=%s param=%g n_rep=%d seed=%d final=%.6g %.1fs\n', ...
        func_name, acf, acf_param, n_rep, seed, Y_min_history(end), runtime);
end
