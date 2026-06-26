clc; clearvars; close all
%%
addpath('../LVGP_Matlab_codes');

%% set seed for random number generator
rng(1);

%% Create problem and choose settings
ind_qual = 2;                 % index of qualitative variable
n_lv = 5;                     % number of levels
lb = -5; ub = 10;             % x1 in [-5,10]
X_range_continuous = [lb; ub]; % bounds for quantitative vars
n_tr_lv = 3;                  % number of unique training locations for each level
n_rep = 10;                   % number of stochastic replicates at each location
n_tr = n_lv*n_tr_lv;          % total number of unique training locations
X_sample = zeros(n_tr, 2);    % unique input samples; col 2 is qualitative level index
Y_sample = zeros(n_tr, 1);    % sample mean at each unique input
Var_sample = zeros(n_tr, 1);  % sample variance at each unique input
Y_rep_sample = cell(n_tr, 1); % optional storage of all replicate observations
var_fctr = [15,2,8,0,10];     % actual values for qualitative variable (level 1 to level 5)
assert(n_lv == length(var_fctr), 'number of levels is incorrect!');
lhs_iter = 1000;

% Test function (1 qualitative and 1 quantitative variable)
f_handle = @(X) (X(:,2)-5.1/4/pi^2*X(:,1).^2+5/pi*X(:,1)-6).^2 ...
                + 10*(1-1/8/pi)*cos(X(:,1))+10;

% Heteroscedastic noise standard deviation. Tune this to match your problem.
nexp = 2;
base_sigma = @(x1) 0.135 .* exp((0.15 .* x1).^nexp);
noise_muls = [1.00, 0.70, 0.90, 0.50, 1.20]*10;
level2mul = @(x2_actual) arrayfun(@(v) noise_muls(find(var_fctr==v, 1, 'first')), x2_actual);
sigma_handle = @(X_actual) base_sigma(X_actual(:,1)) .* level2mul(X_actual(:,2));
obj_noisy_handle = @(X_actual) f_handle(X_actual) + randn(size(X_actual,1),1).*sigma_handle(X_actual);

% Prepare replicated samples for all levels
row = 0;
for i = 1:n_lv
    A = lhsdesign(n_tr_lv, 1, 'iterations', lhs_iter);
    lhs_temp = A.*(ub-lb) + lb;
    for j = 1:n_tr_lv
        row = row + 1;
        x_lvgp = [lhs_temp(j), i];                 % LVGP uses level index
        x_eval = [lhs_temp(j), var_fctr(i)];        % objective uses actual level value
        y_rep = obj_noisy_handle(repmat(x_eval, n_rep, 1));

        X_sample(row,:) = x_lvgp;
        Y_sample(row,1) = mean(y_rep);
        Var_sample(row,1) = var(y_rep, 0, 1);      % unbiased sample variance
        Y_rep_sample{row} = y_rep;
    end
end

acf = 'haei';        % options: ei, lcb, pi, haei, gf
% acf = 'rahbo';        % options: ei, lcb, pi, haei, rahbo, gf
% acf = 'anpei';
% acf = 'gf_new';        % options: ei, lcb, pi, haei, gf
num_iter = 30;       % no. of BO iterations
custom_points = 0;

model_options.ind_qual = ind_qual;  % LVGP model fitting requires index of qualitative vars
model_options.dim_z = 2;            % dimension of latent space

bo_options.n_rep = n_rep;
if strcmp(acf,'haei')
    bo_options.gamma = 1;               % HAEI penalty parameter
end
bo_options.poly_degree = 2;         % polynomial degree for aleatoric variance model
bo_options.poly_lambda = 1e-3;      % ridge regularization
if strcmp(acf,'rahbo')
    bo_options.alpha = 1;     % RAHBO risk-aversion penalty
    bo_options.beta = 2;      % UCB/LCB exploration parameter, if used in acquisition_func
end
if strcmp(acf,'anpei')
    bo_options.beta_anpei = 0.5;
end
bo_options.PlotOrNot = 'Plot';
bo_options.plot_folder = 'LVGP_BO_plots_HAEI';

%% BO
tic
result = bayesian_optimizer(obj_noisy_handle, var_fctr, X_sample, Y_sample, Var_sample, Y_rep_sample, ...
    X_range_continuous, acf, num_iter, custom_points, model_options, bo_options);
elapsed_time = toc;

save(fullfile(bo_options.plot_folder,'data.mat'));

figure
plot(result.Y_min_history,'k-o');
xlabel('Iteration'); ylabel('Best Sample-Mean Function Value y^*')
