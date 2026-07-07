function standard_driver(func_name, acf, acf_param, n_rep, seed, num_iter, out_file)
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
    ind_qual = 2; X_range_continuous = [lb; ub]; n_tr_lv = 2;
    var_fctr = 1:n_lv;
    obj_noisy = @(X) f(X(:,1), X(:,2)) + randn(size(X,1),1) .* sig(X(:,1), X(:,2));

    edge_buf = 1/6; lo = lb + edge_buf*(ub-lb); hi = ub - edge_buf*(ub-lb);
    A = lhsdesign(n_tr_lv, 1, 'iterations', 1000); lhs_shared = A.*(hi-lo) + lo;
    n_tr = n_lv*n_tr_lv;
    X_sample = zeros(n_tr,2); Y_sample = zeros(n_tr,1); Var_sample = zeros(n_tr,1);
    row = 0;
    for i = 1:n_lv
        for j = 1:n_tr_lv
            row = row + 1; xl = [lhs_shared(j), i];
            y_rep = obj_noisy(repmat(xl, n_rep, 1));
            X_sample(row,:) = xl; Y_sample(row) = mean(y_rep); Var_sample(row) = var(y_rep,0,1);
        end
    end

    model_options.ind_qual = ind_qual; model_options.dim_z = 2;
    custom_points = 0; n_points = 2000*size(X_sample,2);

    X_sampled = X_sample; Y_sampled = Y_sample; Yvar_sampled = Var_sample;
    y_min = min(Y_sampled);
    Y_min_history = zeros(1,num_iter); Y_min_est = zeros(1,num_iter);
    X_min_est = zeros(num_iter,2); X_next_history = zeros(num_iter,2);
    Y_next_history = zeros(1,num_iter); Y_var_next_history = zeros(1,num_iter);

    t0 = tic;
    for it = 1:num_iter
        model = LVGP_fit(X_sampled, Y_sampled, model_options);        % MEAN only (homoscedastic)
        [x_next, ~, x_min_est, y_min_est] = find_next(model, X_range_continuous, ...
            acf, n_points, custom_points, X_sampled, Y_sampled);
        x_eval = [x_next(1), var_fctr(int32(x_next(2)))];             % = [x1, level] (identity)
        y_rep  = obj_noisy(repmat(x_eval, n_rep, 1));
        y_mean = mean(y_rep); y_var = var(y_rep,0,1);
        X_sampled = [X_sampled; x_next]; Y_sampled = [Y_sampled; y_mean]; Yvar_sampled = [Yvar_sampled; y_var];
        X_next_history(it,:) = x_next; Y_next_history(it) = y_mean; Y_var_next_history(it) = y_var;
        X_min_est(it,:) = x_min_est; Y_min_est(it) = y_min_est;
        y_min = min(y_min, y_mean); Y_min_history(it) = y_min;
    end
    runtime = toc(t0);

    [~, bi] = min(Y_sampled);
    X_best_final = X_sampled(bi,:); Y_best_final = Y_sampled(bi); Y_var_best_final = Yvar_sampled(bi);
    n_initial = size(X_sample,1);
    Y_min_history = Y_min_history(:)'; Y_sampled = Y_sampled(:)'; Y_var_sampled = Yvar_sampled(:)';
    Y_next_history = Y_next_history(:)'; Y_var_next_history = Y_var_next_history(:)'; Y_min_est = Y_min_est(:)';
    meta = struct('problem',func_name,'model','standard_LVGP','acf',acf,'acf_param',acf_param, ...
                  'n_rep',n_rep,'seed',seed,'num_iter',num_iter,'runtime',runtime);
    save(out_file, 'Y_min_history','X_sampled','Y_sampled','Y_var_sampled', ...
        'X_next_history','Y_next_history','Y_var_next_history','Y_min_est','X_min_est', ...
        'X_best_final','Y_best_final','Y_var_best_final','n_initial','var_fctr','meta','-v7');
    fprintf('DONE %s standard_LVGP acf=%s n_rep=%d seed=%d final=%.6g %.1fs\n', ...
        func_name, acf, n_rep, seed, Y_min_history(end), runtime);
end
