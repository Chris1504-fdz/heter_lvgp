function [x_next, U_min_est, x_min_est, y_min_est] = find_next(model, X_range_continuous, acf, n_points, custom_points, X_sample, y_sample)

if custom_points == 0
    d = size(X_sample,2);
    dim_qual = model.data.ind_qual; d_quan = d - length(dim_qual);
    levels = model.data.n_lvs_qual;
    n_top = 4*d;

    lb = X_range_continuous(1,:); ub = X_range_continuous(2,:);

    % sample n_points in feasible space
    X_pred = lhsdesign(n_points, d, 'iterations', 100); 
    X_pred(:, 1:d_quan) = lb + X_pred(:, 1:d_quan) .* (ub-lb);
    X_pred(:, dim_qual) = ceil(repmat(levels, [n_points,1]).*X_pred(:, dim_qual));
    y = LVGP_predict(X_pred,model,'MSE_on', true);
    y_pred = y.Y_hat; y_cov = y.MSE;
    s = sqrt(abs(diag(y_cov)));

    options = optimoptions('fmincon','Display','off');
    %% select the best n_top points for small mean
    [~, index] = sort(y_pred);
    x_top_mean = X_pred(index(1:n_top),:);

    % use the best n_top points as initial points for local search
    x_cont_opt_set = zeros(n_top, d_quan);
    y_opt_set = zeros(n_top,1);
    parfor i = 1:n_top
        problem1 = createOptimProblem('fmincon', 'lb', lb, 'ub', ub, 'x0', x_top_mean(i, 1:d_quan),...
            'objective',@(X_continuous) LVGP_predict([X_continuous, x_top_mean(i, dim_qual)], model).Y_hat, 'options', options);
        [x_cont_opt_set(i,:), y_opt_set(i)] = fmincon(problem1);
    end

    [y_min_est, min_idx] = min(y_opt_set);
    x_min_est = [x_cont_opt_set(min_idx,:), x_top_mean(min_idx, dim_qual)];

    %% find the optimizer of acquisition function
    U_negate = acquisition_func(model, X_pred, min(y_sample), acf, y_pred, s);
    [~, index_U] = sort(U_negate);
    x_top_U = X_pred(index_U(1:n_top),:);

    x_cont_opt_U_set = zeros(n_top, d_quan);
    U_opt_set = zeros(n_top,1);
    parfor j = 1:n_top
        problem2 = createOptimProblem('fmincon', 'lb', lb, 'ub', ub, 'x0', x_top_U(j, 1:d_quan),...
            'objective',@(X_continuous) acquisition_func(model, [X_continuous, x_top_U(j, dim_qual)], y_min_est, acf), 'options', options);
        [x_cont_opt_U_set(j,:), U_opt_set(j)] = fmincon(problem2);
    end

    [U_min_est, min_U_idx] = min(U_opt_set);
    x_next = [x_cont_opt_U_set(min_U_idx,:), x_top_U(min_U_idx, dim_qual)];

elseif size(custom_points,1)>1

    custom_points = setdiff(custom_points, X_sample, 'rows');
    y = LVGP_predict(custom_points, model,'MSE_on', true);
    y_pred = y.Y_hat; y_cov = y.MSE;
    s = sqrt(abs(diag(y_cov)));
    [y_min_est, min_idx] = min(y_pred);
    x_min_est = custom_points(min_idx,:);
    
    U_negate = acquisition_func(model, custom_points, min(y_sample), acf, y_pred, s);
    [U_min_est, min_U_idx] = min(U_negate);
    x_next = custom_points(min_U_idx,:);
    
%     figure; 
%     subplot(3,1,1); plot(y_pred);
%     subplot(3,1,2); plot(s);
%     subplot(3,1,3); plot(-U_negate);
    
end
