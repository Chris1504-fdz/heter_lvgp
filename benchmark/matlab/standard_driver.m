function standard_driver(func_name, acf, acf_param, n_rep, seed, num_iter, out_file, doe_file)
% Standard (homoscedastic, noise-UNAWARE) LVGP engine (generalizes study_v2_plain_lvgp/study_driver.m).
% Fits the plain LVGP on the replicate MEAN only (variance recorded but never modelled); supports
% only ei/lcb/pi. Builds the problem from problems.m; var_fctr = 1:n_lv (objective gets [x1, level]).

    here = fileparts(mfilename('fullpath'));
    addpath(here);                                   % problems.m
    addpath(fullfile(here, 'standard_lvgp'));        % BO_standard_LVGP (incl my_grad.m)
    try, pset = parallel.Settings; pset.Pool.AutoCreate = false; catch, end
    if ~any(strcmpi(acf, {'ei','lcb','pi'}))
        error('standard_LVGP supports only ei/lcb/pi (noise-unaware); got "%s"', acf);
    end
    rng(seed);

    [f, sig, lb, ub, n_lv] = problems(func_name);
    nd = numel(lb);                                  % # continuous dims (lb/ub are 1 x nd rows)
    ind_qual = nd + 1; X_range_continuous = [lb; ub];% DOE (incl. its size) comes from doe_file
    var_fctr = 1:n_lv;
    obj_noisy = @(X) f(X(:,1:nd), X(:,nd+1)) + randn(size(X,1),1) .* sig(X(:,1:nd), X(:,nd+1));

    % ---- initial DOE: SHARED design generated in Python (utils/doe_cache.py) -- common random numbers,
    %      byte-identical for every model. The plain LVGP uses only the replicate MEAN. ----
    D = load(doe_file);
    X_sample = D.X_sample; Y_sample = D.Y_sample(:); Var_sample = D.Var_sample(:);
    n_tr = size(X_sample, 1);

    model_options.ind_qual = ind_qual; model_options.dim_z = 2;
    custom_points = 0; n_points = 2000*size(X_sample,2);

    X_sampled = X_sample; Y_sampled = Y_sample; Yvar_sampled = Var_sample;
    y_min = min(Y_sampled);
    Y_min_history = zeros(1,num_iter); Y_min_est = zeros(1,num_iter);
    X_min_est = zeros(num_iter,nd+1); X_next_history = zeros(num_iter,nd+1);
    Y_next_history = zeros(1,num_iter); Y_var_next_history = zeros(1,num_iter);
    acf_val = zeros(1,num_iter);
    mu_at_est = zeros(1,num_iter); s_at_est = zeros(1,num_iter);
    r_at_est = nan(1,num_iter);            % NaN by construction: this model is noise-UNAWARE
    Y_rep_sampled = [D.Y_rep; zeros(num_iter, n_rep)];   % initial block, then one row per iteration

    t0 = tic;
    for it = 1:num_iter
        model = LVGP_fit(X_sampled, Y_sampled, model_options);        % MEAN only (homoscedastic)
        [x_next, U_min_est, x_min_est, y_min_est] = find_next(model, X_range_continuous, ...
            acf, n_points, custom_points, X_sampled, Y_sampled);
        x_eval = [x_next(1:end-1), var_fctr(int32(x_next(end)))];     % = [x_1..x_nd, level] (identity)
        y_rep  = obj_noisy(repmat(x_eval, n_rep, 1));
        y_mean = mean(y_rep); y_var = var(y_rep,0,1);
        X_sampled = [X_sampled; x_next]; Y_sampled = [Y_sampled; y_mean]; Yvar_sampled = [Yvar_sampled; y_var];
        X_next_history(it,:) = x_next; Y_next_history(it) = y_mean; Y_var_next_history(it) = y_var;
        X_min_est(it,:) = x_min_est; Y_min_est(it) = y_min_est;
        y_min = min(y_min, y_mean); Y_min_history(it) = y_min;

        % ---- v2 history: acquisition value + posterior at the recommended optimum ----
        acf_val(it) = -U_min_est;                                     % same sign convention as heter
        pred_est = LVGP_predict(x_min_est(:)', model, 'MSE_on', true);
        mu_at_est(it) = pred_est.Y_hat(1);
        s_at_est(it)  = sqrt(max(pred_est.MSE(1), 0));                % epistemic std
        Y_rep_sampled(n_tr + it, :) = y_rep(:)';       % n_tr = size(X_sample,1), the initial block
    end
    runtime = toc(t0);

    [~, bi] = min(Y_sampled);
    X_best_final = X_sampled(bi,:); Y_best_final = Y_sampled(bi); Y_var_best_final = Yvar_sampled(bi);
    n_initial = size(X_sample,1);
    Y_min_history = Y_min_history(:)'; Y_sampled = Y_sampled(:)'; Y_var_sampled = Yvar_sampled(:)';
    Y_next_history = Y_next_history(:)'; Y_var_next_history = Y_var_next_history(:)'; Y_min_est = Y_min_est(:)';

    % ---- SCHEMA v2 ----
    X_init = X_sample; Y_init = D.Y_sample(:)'; Y_rep_init = D.Y_rep;
    f_true_sampled     = reshape(f(X_sampled(:,1:nd),   X_sampled(:,nd+1)), 1, []);
    sigma_true_sampled = reshape(sig(X_sampled(:,1:nd), X_sampled(:,nd+1)), 1, []);

    % curated hyperparameters. The plain LVGP is homoscedastic -> no aleatoric polynomial exists.
    hyper = struct();
    hyper.z        = model.qual_param.z;                % latent embedding of the categorical levels
    hyper.phi      = model.quant_param.phi;             % continuous lengthscales
    hyper.beta_hat = model.fit_detail.beta_hat;
    hyper.sigma2   = model.fit_detail.sigma2;
    hyper.nug_opt  = model.fit_detail.nug_opt;

    meta = struct('problem',func_name,'model','standard_LVGP','acf',acf,'acf_param',acf_param, ...
                  'n_rep',n_rep,'seed',seed,'num_iter',num_iter,'runtime',runtime, ...
                  'schema_version',2,'doe_mode','slhd','n_init',n_initial,'n_levels',n_lv, ...
                  'lb',lb,'ub',ub,'timestamp',datestr(now,'yyyy-mm-ddTHH:MM:SS'), ...
                  'noise_stream','matlab:rng(seed)');

    % ATOMIC write (see heter_driver.m): temp file in the destination dir, then rename over target.
    [out_dir, out_base] = fileparts(out_file);
    tmp_file = fullfile(out_dir, sprintf('%s.tmp%d.mat', out_base, feature('getpid')));
    save(tmp_file, 'Y_min_history','X_sampled','Y_sampled','Y_var_sampled', ...
        'X_next_history','Y_next_history','Y_var_next_history','Y_min_est','X_min_est', ...
        'X_best_final','Y_best_final','Y_var_best_final','n_initial','var_fctr','meta', ...
        'Y_rep_sampled','acf_val','mu_at_est','s_at_est','r_at_est', ...
        'X_init','Y_init','Y_rep_init','f_true_sampled','sigma_true_sampled','hyper','-v7');
    movefile(tmp_file, out_file, 'f');
    fprintf('DONE %s standard_LVGP acf=%s n_rep=%d seed=%d final=%.6g %.1fs\n', ...
        func_name, acf, n_rep, seed, Y_min_history(end), runtime);
end
