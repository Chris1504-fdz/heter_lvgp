function heter_driver(func_name, acf, acf_param, n_rep, seed, num_iter, out_file)
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
    ind_qual = 2; X_range_continuous = [lb; ub]; n_tr_lv = 2;
    var_fctr = 1:n_lv;                               % identity -> objfunc gets [x1, level]
    obj_noisy = @(X) f(X(:,1), X(:,2)) + randn(size(X,1),1) .* sig(X(:,1), X(:,2));

    % ---- initial DOE: maximin LHS on a 1/6 inset, shared x1 across levels ----
    edge_buf = 1/6; lo = lb + edge_buf*(ub-lb); hi = ub - edge_buf*(ub-lb);
    A = lhsdesign(n_tr_lv, 1, 'iterations', 1000); lhs_shared = A.*(hi-lo) + lo;
    n_tr = n_lv*n_tr_lv;
    X_sample = zeros(n_tr,2); Y_sample = zeros(n_tr,1); Var_sample = zeros(n_tr,1); Y_rep_sample = cell(n_tr,1);
    row = 0;
    for i = 1:n_lv
        for j = 1:n_tr_lv
            row = row + 1; xl = [lhs_shared(j), i];
            y_rep = obj_noisy(repmat(xl, n_rep, 1));
            X_sample(row,:) = xl; Y_sample(row) = mean(y_rep);
            Var_sample(row) = var(y_rep,0,1); Y_rep_sample{row} = y_rep;
        end
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
    meta = struct('problem',func_name,'model','heter_LVGP','acf',acf,'acf_param',acf_param, ...
                  'n_rep',n_rep,'seed',seed,'num_iter',num_iter,'runtime',runtime);
    save(out_file, 'Y_min_history','X_sampled','Y_sampled','Y_var_sampled', ...
        'X_next_history','Y_next_history','Y_var_next_history','Y_min_est','X_min_est', ...
        'X_best_final','Y_best_final','Y_var_best_final','n_initial','var_fctr','meta','-v7');
    fprintf('DONE %s heter_LVGP acf=%s param=%g n_rep=%d seed=%d final=%.6g %.1fs\n', ...
        func_name, acf, acf_param, n_rep, seed, Y_min_history(end), runtime);
end
