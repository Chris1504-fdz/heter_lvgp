function result = bayesian_optimizer(objfunc, var_fctr, X_sample, y_sample,...
    y_var_sample, y_rep_sampled, X_range_continuous, acf, num_iter, custom_points, model_options, bo_options)
% Heteroscedastic LVGP Bayesian optimizer for stochastic objectives.
%
% Required data format:
%   X_sample      : unique design locations. Qualitative variables are level indices.
%   y_sample      : sample mean at each unique location.
%   y_var_sample  : sample variance from replicates at each unique location.
%   objfunc       : stochastic objective handle evaluated using actual qualitative values.
%
% This version uses LVGP_fit_noise and LVGP_predict_noise and refits a
% polynomial aleatoric-variance model after each BO iteration.

if nargin < 11 || isempty(bo_options)
    bo_options = struct();
end
if ~isfield(bo_options, 'n_rep'), bo_options.n_rep = 10; end
if ~isfield(bo_options, 'gamma'), bo_options.gamma = 1; end
if ~isfield(bo_options, 'poly_degree'), bo_options.poly_degree = 2; end
if ~isfield(bo_options, 'poly_lambda'), bo_options.poly_lambda = 1e-3; end
if ~isfield(bo_options, 'PlotOrNot'), bo_options.PlotOrNot = 'DontPlot'; end
if ~isfield(bo_options, 'plot_folder'), bo_options.plot_folder = 'LVGP_BO_plots'; end

% Histories are based on the sample mean, not on one noisy observation.
y_min = min(y_sample);
U = zeros(num_iter, 1);
Y_min = zeros(num_iter, 1); Y_min_est = Y_min;
X_min_est = zeros(num_iter, size(X_sample, 2));
X_sampled = X_sample;
y_sampled = y_sample;
y_var_sampled = y_var_sample;
% y_rep_sampled = cell(size(X_sample,1),1);

n_points = 2000*size(X_sample,2);
PlotOrNot = bo_options.PlotOrNot;
plot_folder = bo_options.plot_folder;

if strcmp(PlotOrNot,'Plot') && ~exist(plot_folder,'dir')
    mkdir(plot_folder);
end

X_next_history = zeros(num_iter, size(X_sample, 2));
Y_next_history = zeros(num_iter, 1);
Y_var_next_history = zeros(num_iter, 1);
Y_rep_next_history = cell(num_iter, 1);

for i = 1:num_iter

    % Heteroscedastic LVGP model fitting using sample means and variances.
    model = LVGP_fit_noise(X_sampled, y_sampled, y_var_sampled, model_options);

    % Fit polynomial model r(x) for aleatoric variance at unknown locations.
    model.aleatoric_poly = fit_aleatoric_polymodel(X_sampled, y_var_sampled, model, bo_options);
    model.y_var_sample = y_var_sampled;
    model.bo_options = bo_options;

    % Generating plots
    if strcmp(PlotOrNot,'Plot')==1
        x_grid = linspace(min(X_sampled(:,1)), max(X_sampled(:,1)), 100)';
        figure('Position', [100, 100, 1200, 400]);
        tiledlayout(2,3);

        for level = 1:length(var_fctr)
            nexttile;
            idx = (X_sampled(:,2) == level);
            x_data = X_sampled(idx, 1);
            y_data = y_sampled(idx);

            X_pred = [x_grid, level * ones(size(x_grid))];
            output = LVGP_predict_noise(X_pred, y_var_sampled, model, 'MSE_on', true);
            y_pred = output.Y_hat(:);
            epi_var = diag(output.MSE);
            ale_var = predict_aleatoric_variance_local(X_pred, model);
            y_std = sqrt(max(epi_var + ale_var, 0));

            fill([x_grid; flipud(x_grid)], ...
                 [y_pred - 1.96*y_std; flipud(y_pred + 1.96*y_std)], ...
                 [0.8 0.8 1], 'EdgeColor', 'none', 'FaceAlpha', 0.5); hold on;
            plot(x_grid, y_pred, 'b-', 'LineWidth', 1.5);
            % plot(x_data, y_data, 'ko', 'MarkerFaceColor', 'k');
            
            % Plot all replicate observations for this categorical level
            rep_cells = y_rep_sampled(idx);   % cells corresponding to this level
            
            for k = 1:numel(rep_cells)
                y_rep_k = rep_cells{k}(:);
                x_rep_k = repmat(x_data(k), size(y_rep_k));
            
                scatter(x_rep_k, y_rep_k, 20, ...
                    'MarkerFaceColor', 'g', ...
                    'MarkerEdgeColor', 'k', ...
                    'MarkerFaceAlpha', 0.8);
            end

            title(['Category Level ', num2str(level)]);
            xlabel('x'); ylabel('y'); grid on;
        end
    end

    % Find min and next sampling point.
    [x_next, U_min_est, x_min_est, y_min_est] = find_next(model, X_range_continuous, acf, ...
        n_points, custom_points, X_sampled, y_sampled, y_var_sampled, bo_options);

    % Evaluate the selected location with n_rep stochastic replicates.
    x_next_eval = [x_next(1), var_fctr(int32(x_next(2)))];
    y_rep_next = objfunc(repmat(x_next_eval, bo_options.n_rep, 1));
    y_next = mean(y_rep_next);
    y_var_next = var(y_rep_next, 0, 1);

    X_next_history(i,:) = x_next;
    Y_next_history(i) = y_next;
    Y_var_next_history(i) = y_var_next;
    Y_rep_next_history{i} = y_rep_next;

    if strcmp(PlotOrNot,'Plot')==1
        nexttile(int32(x_next(2)));
        % plot(x_next(1), y_next, 'r*', 'MarkerSize', 8);
        % Plot replicate observations for the new sampled point
        x_rep_next = repmat(x_next(1), size(y_rep_next));
        scatter(x_rep_next, y_rep_next, 30, ...
            'MarkerFaceColor', 'r', ...
            'MarkerEdgeColor', 'k', ...
            'MarkerFaceAlpha', 0.9);
        sgtitle(sprintf('Heteroscedastic LVGP BO - Iteration %d', i));
        saveas(gcf, fullfile(plot_folder, sprintf('figure_iter_%03d.png', i)));
        close(gcf);
    end

    y_min = min(y_next, y_min);

    X_sampled = [X_sampled; x_next];
    y_sampled = [y_sampled; y_next];
    y_var_sampled = [y_var_sampled; y_var_next];
    y_rep_sampled{end+1,1} = y_rep_next;

    Y_min(i) = y_min;
    Y_min_est(i) = y_min_est;
    X_min_est(i,:) = x_min_est;
    U(i) = -U_min_est;

    fprintf('Iteration %u completed. ', i)
    fprintf('Current sample-mean minimum is %f. New variance estimate is %f.\n', y_min, y_var_next);
end

result.Y_optimum = y_min;
result.Y_min_history = Y_min;
result.X_sampled = X_sampled;
result.Y_sampled = y_sampled;
result.Y_var_sampled = y_var_sampled;
result.Y_rep_sampled = y_rep_sampled;
result.acf_val = U;
result.final_model = model;
result.Y_min_est = Y_min_est;
result.X_min_est = X_min_est;

result.X_next_history = X_next_history;
result.Y_next_history = Y_next_history;
result.Y_var_next_history = Y_var_next_history;
result.Y_rep_next_history = Y_rep_next_history;

[best_val, best_idx] = min(y_sampled);
result.Y_best_final = best_val;
result.X_best_final = X_sampled(best_idx,:);
result.Y_var_best_final = y_var_sampled(best_idx);
result.Y_rep_best_final = y_rep_sampled{best_idx};

end

function poly = fit_aleatoric_polymodel(X, y_var, model, bo_options)
% Fit log(sigma) = theta' Phi([x_cont, latent categorical coordinates]).
ind_qual = model.data.ind_qual;
d = size(X,2);
cont_idx = setdiff(1:d, ind_qual);
Z_latent = model.qual_param.z{1};

W = zeros(size(X,1), numel(cont_idx) + size(Z_latent,2));
for ii = 1:size(X,1)
    c = X(ii, ind_qual);
    W(ii,:) = [X(ii, cont_idx), Z_latent(c,:)];
end

mu_W = mean(W, 1);
std_W = std(W, 0, 1);
std_W(std_W == 0) = 1;
Wn = (W - mu_W) ./ std_W;

Phi = build_poly_features_local(Wn, bo_options.poly_degree);
log_sigma = 0.5 * log(max(y_var, 1e-12));
lambda = bo_options.poly_lambda;
theta = (Phi' * Phi + lambda * eye(size(Phi,2))) \ (Phi' * log_sigma);

poly.theta = theta;
poly.degree = bo_options.poly_degree;
poly.lambda = lambda;
poly.mu_W = mu_W;
poly.std_W = std_W;
poly.ind_qual = ind_qual;
poly.cont_idx = cont_idx;
poly.Z_latent = Z_latent;
end

function Phi = build_poly_features_local(Wn, degree)
Phi = ones(size(Wn,1), 1);
for deg = 1:degree
    Phi = [Phi, Wn.^deg]; %#ok<AGROW>
end
if size(Wn,2) >= 2
    for a = 1:size(Wn,2)-1
        for b = a+1:size(Wn,2)
            Phi = [Phi, Wn(:,a).*Wn(:,b)]; %#ok<AGROW>
        end
    end
end
end

function var_pred = predict_aleatoric_variance_local(Xnew, model)
poly = model.aleatoric_poly;
Wnew = zeros(size(Xnew,1), numel(poly.cont_idx) + size(poly.Z_latent,2));
for ii = 1:size(Xnew,1)
    c = Xnew(ii, poly.ind_qual);
    Wnew(ii,:) = [Xnew(ii, poly.cont_idx), poly.Z_latent(c,:)];
end
Wn = (Wnew - poly.mu_W) ./ poly.std_W;
Phi = build_poly_features_local(Wn, poly.degree);
log_sigma = Phi * poly.theta;
var_pred = exp(2*log_sigma);
var_pred = max(var_pred, 1e-12);
end
