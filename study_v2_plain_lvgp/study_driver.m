function study_driver(acf, acf_param, n_rep, seed, num_iter, out_file)
% Standard (homoscedastic) LVGP Bayesian-optimization driver -- the NOISE-UNAWARE baseline.
%
% Same test problem / DOE / saved .mat schema as ../study_v2/study_driver.m (so utils/results.py
% and the comparison notebooks load it directly), but:
%   * the surrogate is the plain LVGP in BO_standard_LVGP/ (one global noise nugget, no
%     heteroscedastic / aleatoric model), called with its functions UNMODIFIED;
%   * the model is fit on the replicate MEAN ONLY -- the replicate variance is recorded (for the
%     analysis plots) but NEVER passed to LVGP_fit / the acquisition; and
%   * only ei / lcb / pi are supported (haei/anpei/rahbo need an aleatoric r(x) this study omits).
%
%   acf       : 'ei' | 'lcb' | 'pi'
%   acf_param : unused (pass NaN) -- kept for a uniform call signature with study_v2
%   n_rep     : replicates per evaluated location (3|5|10); only their MEAN reaches the model
%   seed      : RNG seed (controls initial DOE + the noise stream)
%   num_iter  : number of BO iterations
%   out_file  : .mat path to save results

    here = fileparts(mfilename('fullpath'));
    addpath(fullfile(here, 'BO_standard_LVGP'));   % LVGP_fit / LVGP_predict / find_next / acquisition_func

    % find_next.m uses parfor unconditionally; we parallelize at the Python level, so force those
    % loops to run serially in this single process (does NOT modify their files).
    try
        pset = parallel.Settings;
        pset.Pool.AutoCreate = false;
    catch
    end

    if ~any(strcmpi(acf, {'ei','lcb','pi'}))
        error('study_v2_plain_lvgp supports only ei/lcb/pi (noise-unaware); got "%s"', acf);
    end
    rng(seed);

    %% ---- problem definition (identical to study_v2/study_driver.m) ----
    ind_qual = 2;  n_lv = 5;  lb = -5;  ub = 10;
    X_range_continuous = [lb; ub];
    n_tr_lv = 2;                          % 2 LHS points per category (v2 DOE)
    var_fctr = [15,2,8,0,10];

    f_handle = @(X) (X(:,2)-5.1/4/pi^2*X(:,1).^2+5/pi*X(:,1)-6).^2 ...
                    + 10*(1-1/8/pi)*cos(X(:,1))+10;
    nexp = 2;
    base_sigma  = @(x1) 0.135 .* exp((0.15 .* x1).^nexp);
    noise_muls  = [1.00, 0.70, 0.90, 0.50, 1.20]*10;
    level2mul   = @(x2_actual) arrayfun(@(v) noise_muls(find(var_fctr==v,1,'first')), x2_actual);
    sigma_handle= @(X_actual) base_sigma(X_actual(:,1)) .* level2mul(X_actual(:,2));
    obj_noisy   = @(X_actual) f_handle(X_actual) + randn(size(X_actual,1),1).*sigma_handle(X_actual);

    %% ---- initial DOE: maximin LHS over a 1/6 inset, SHARED across all categories ----
    edge_buf = 1/6;
    lo = lb + edge_buf*(ub - lb);                    % -2.5
    hi = ub - edge_buf*(ub - lb);                    %  7.5
    A = lhsdesign(n_tr_lv, 1, 'iterations', 1000);   % maximin
    lhs_shared = A.*(hi - lo) + lo;

    n_tr = n_lv*n_tr_lv;
    X_sample = zeros(n_tr, 2);   Y_sample = zeros(n_tr, 1);   Var_sample = zeros(n_tr, 1);
    row = 0;
    for i = 1:n_lv
        for j = 1:n_tr_lv
            row = row + 1;
            X_sample(row,:) = [lhs_shared(j), i];                       % LVGP coding [x1, level idx]
            y_rep = obj_noisy(repmat([lhs_shared(j), var_fctr(i)], n_rep, 1));
            Y_sample(row)   = mean(y_rep);                              % MEAN -> the model
            Var_sample(row) = var(y_rep, 0, 1);                         % variance RECORDED ONLY
        end
    end

    %% ---- options ----
    model_options.ind_qual = ind_qual;
    model_options.dim_z = 2;
    custom_points = 0;
    n_points = 2000*size(X_sample, 2);

    %% ---- BO loop (standard LVGP; the model sees the MEAN only) ----
    X_sampled = X_sample;  Y_sampled = Y_sample;  Yvar_sampled = Var_sample;
    y_min = min(Y_sampled);
    Y_min_history = zeros(1, num_iter);  Y_min_est = zeros(1, num_iter);
    X_min_est = zeros(num_iter, 2);      X_next_history = zeros(num_iter, 2);
    Y_next_history = zeros(1, num_iter);  Y_var_next_history = zeros(1, num_iter);

    t0 = tic;
    for it = 1:num_iter
        model = LVGP_fit(X_sampled, Y_sampled, model_options);          % homoscedastic, MEAN only
        [x_next, ~, x_min_est, y_min_est] = find_next(model, X_range_continuous, ...
            acf, n_points, custom_points, X_sampled, Y_sampled);

        x_eval = [x_next(1), var_fctr(int32(x_next(2)))];
        y_rep  = obj_noisy(repmat(x_eval, n_rep, 1));
        y_mean = mean(y_rep);   y_var = var(y_rep, 0, 1);

        X_sampled   = [X_sampled; x_next];        Y_sampled = [Y_sampled; y_mean];
        Yvar_sampled= [Yvar_sampled; y_var];
        X_next_history(it,:)    = x_next;         Y_next_history(it) = y_mean;
        Y_var_next_history(it)  = y_var;
        X_min_est(it,:)         = x_min_est;      Y_min_est(it) = y_min_est;
        y_min = min(y_min, y_mean);               Y_min_history(it) = y_min;
    end
    runtime = toc(t0);

    %% ---- final best OBSERVED design, then save (study_v2 schema) ----
    [~, bi] = min(Y_sampled);
    X_best_final     = X_sampled(bi, :);
    Y_best_final     = Y_sampled(bi);
    Y_var_best_final = Yvar_sampled(bi);
    n_initial        = size(X_sample, 1);

    Y_min_history = Y_min_history(:)';
    Y_sampled     = Y_sampled(:)';
    Y_var_sampled = Yvar_sampled(:)';
    Y_next_history     = Y_next_history(:)';
    Y_var_next_history = Y_var_next_history(:)';
    Y_min_est          = Y_min_est(:)';

    meta = struct('acf',acf,'acf_param',acf_param,'n_rep',n_rep, ...
                  'seed',seed,'num_iter',num_iter,'runtime',runtime,'model','plain_lvgp');

    save(out_file, 'Y_min_history','X_sampled','Y_sampled','Y_var_sampled', ...
        'X_next_history','Y_next_history','Y_var_next_history', ...
        'Y_min_est','X_min_est','X_best_final','Y_best_final','Y_var_best_final', ...
        'n_initial','var_fctr','meta', '-v7');

    fprintf('DONE acf=%s param=%g n_rep=%d seed=%d  final_y=%.6g  %.1fs\n', ...
        acf, acf_param, n_rep, seed, Y_min_history(end), runtime);
end
