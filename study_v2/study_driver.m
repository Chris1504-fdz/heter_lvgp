function study_driver(acf, acf_param, n_rep, seed, num_iter, out_file)
% Thin driver for the acquisition-function / n_rep parameter study.
% Calls the postdoc's UNMODIFIED functions (bayesian_optimizer, LVGP_*).
% Rebuilds the same test problem as MAIN_NEW.m, parameterized.
%
%   acf       : 'ei' | 'lcb' | 'pi' | 'haei' | 'anpei' | 'rahbo'
%   acf_param : knob value -> haei:gamma, anpei:beta_anpei, rahbo:alpha
%               (ignored for ei/lcb/pi; pass NaN)
%   n_rep     : stochastic replicates per training location (3|5|10)
%   seed      : RNG seed (controls initial DOE + noise stream)
%   num_iter  : number of BO iterations
%   out_file  : .mat path to save results

    here = fileparts(mfilename('fullpath'));
    addpath(fullfile(here, '..', 'LVGP_Matlab_codes'));
    addpath(fullfile(here, '..', 'Heter_BO_GF'));

    % The postdoc's find_next.m uses parfor unconditionally, which auto-starts
    % an ~18-worker process pool per run (huge memory). We parallelize at the
    % outer (Python) level instead, so force those parfor loops to run serially
    % in this single process. (Does NOT modify their files.)
    try
        pset = parallel.Settings;
        pset.Pool.AutoCreate = false;
    catch
    end

    rng(seed);

    %% ---- problem definition (same as MAIN_NEW.m) ----
    ind_qual = 2;  n_lv = 5;  lb = -5; ub = 10;
    X_range_continuous = [lb; ub];
    n_tr_lv = 2;                 % v2: 2 LHS points per category (was 3)
    n_tr = n_lv*n_tr_lv;
    X_sample = zeros(n_tr, 2);
    Y_sample = zeros(n_tr, 1);
    Var_sample = zeros(n_tr, 1);
    Y_rep_sample = cell(n_tr, 1);
    var_fctr = [15,2,8,0,10];

    f_handle = @(X) (X(:,2)-5.1/4/pi^2*X(:,1).^2+5/pi*X(:,1)-6).^2 ...
                    + 10*(1-1/8/pi)*cos(X(:,1))+10;

    nexp = 2;
    base_sigma = @(x1) 0.135 .* exp((0.15 .* x1).^nexp);
    noise_muls = [1.00, 0.70, 0.90, 0.50, 1.20]*10;
    level2mul = @(x2_actual) arrayfun(@(v) noise_muls(find(var_fctr==v,1,'first')), x2_actual);
    sigma_handle = @(X_actual) base_sigma(X_actual(:,1)) .* level2mul(X_actual(:,2));
    obj_noisy_handle = @(X_actual) f_handle(X_actual) + randn(size(X_actual,1),1).*sigma_handle(X_actual);

    %% ---- initial replicated samples ----
    % v2: maximin LHS over an inset sampling range, generated ONCE and SHARED
    % across all categories, so every level uses the same x1 sample locations
    % (tight, consistent bands across seeds -- like the v1 plot -- rather than
    % dispersing per level). maximin pins the points to the range edges, so the
    % inset both keeps them off the bounds AND sets the band locations. A 1/6
    % inset puts the edges at the CENTERS of the outer thirds of the domain:
    % thirds of [-5,10] = [-5,0],[0,5],[5,10] -> centers -2.5 and 7.5. The two
    % bands sit at x1~-2.5 and ~7.5, bracketing the middle third (which holds
    % the true optimum x1~3.18) and staying off both bounds.
    edge_buf = 1/6;                                  % 1/6 inset each side
    lo = lb + edge_buf*(ub - lb);                    % -2.5  (center of lower third)
    hi = ub - edge_buf*(ub - lb);                    %  7.5  (center of upper third)
    A = lhsdesign(n_tr_lv, 1, 'iterations', 1000);   % maximin (default criterion)
    lhs_shared = A.*(hi - lo) + lo;                  % same x1 values for every level

    row = 0;
    for i = 1:n_lv
        lhs_temp = lhs_shared;
        for j = 1:n_tr_lv
            row = row + 1;
            x_lvgp = [lhs_temp(j), i];
            x_eval = [lhs_temp(j), var_fctr(i)];
            y_rep = obj_noisy_handle(repmat(x_eval, n_rep, 1));
            X_sample(row,:) = x_lvgp;
            Y_sample(row,1) = mean(y_rep);
            Var_sample(row,1) = var(y_rep, 0, 1);
            Y_rep_sample{row} = y_rep;
        end
    end

    %% ---- options ----
    custom_points = 0;
    model_options.ind_qual = ind_qual;
    model_options.dim_z = 2;

    bo_options.n_rep = n_rep;
    bo_options.poly_degree = 2;
    bo_options.poly_lambda = 1e-3;
    bo_options.PlotOrNot = 'DontPlot';          % headless-safe; no figures

    switch lower(acf)
        case 'haei'
            bo_options.gamma = acf_param;
        case 'rahbo'
            bo_options.alpha = acf_param;
            bo_options.beta  = 2;
        case 'anpei'
            bo_options.beta_anpei = acf_param;
        case {'ei','lcb','pi'}
            % no extra knob
        otherwise
            error('Unknown acf: %s', acf);
    end

    %% ---- run BO ----
    t0 = tic;
    result = bayesian_optimizer(obj_noisy_handle, var_fctr, X_sample, Y_sample, ...
        Var_sample, Y_rep_sample, X_range_continuous, acf, num_iter, ...
        custom_points, model_options, bo_options);
    runtime = toc(t0);

    %% ---- save rich results (all numeric -> scipy.io readable) ----
    % Enough to regenerate every plot without re-running the BO.
    Y_min_history      = result.Y_min_history(:)';   % best sample-mean min vs iter
    X_sampled          = result.X_sampled;           % [x1, level_idx] all evals (initial + BO)
    Y_sampled          = result.Y_sampled(:)';       % sample mean at each location
    Y_var_sampled      = result.Y_var_sampled(:)';   % aleatoric (sample) variance at each
    X_next_history     = result.X_next_history;      % BO-chosen point each iteration
    Y_next_history     = result.Y_next_history(:)';
    Y_var_next_history = result.Y_var_next_history(:)';
    Y_min_est          = result.Y_min_est(:)';       % recommended-optimum objective vs iter
    X_min_est          = result.X_min_est;           % recommended-optimum location vs iter
    X_best_final       = result.X_best_final;        % final best observed design [x1, level_idx]
    Y_best_final       = result.Y_best_final;        % its objective
    Y_var_best_final   = result.Y_var_best_final;    % its aleatoric variance
    n_initial          = size(X_sample, 1);          % # initial DOE points (rest are BO)

    meta = struct('acf',acf,'acf_param',acf_param,'n_rep',n_rep, ...
                  'seed',seed,'num_iter',num_iter,'runtime',runtime);

    save(out_file, 'Y_min_history','X_sampled','Y_sampled','Y_var_sampled', ...
        'X_next_history','Y_next_history','Y_var_next_history', ...
        'Y_min_est','X_min_est','X_best_final','Y_best_final','Y_var_best_final', ...
        'n_initial','var_fctr','meta', '-v7');

    fprintf('DONE acf=%s param=%g n_rep=%d seed=%d  final_y=%.6g  %.1fs\n', ...
        acf, acf_param, n_rep, seed, Y_min_history(end), runtime);
end
