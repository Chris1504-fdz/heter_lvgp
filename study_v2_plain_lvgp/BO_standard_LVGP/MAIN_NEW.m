clc; clearvars; close all
%%
addpath('LVGP_Matlab_codes');

%% set seed for randon num. generator 
rng(1);

%% Create problem and choose settings
ind_qual = 2;                 % index of qualitative variable
n_lv = 5;                     % number of levels
lb = -5; ub = 10;             % x1 in [-5,10]
X_range_continuous = [lb;ub]; % bounds for quantitative vars
n_tr_lv = 3;                  % number of training samples for each level
n_tr = n_lv*n_tr_lv;          % total number of training samples
X_sample = zeros(n_tr, 2);    % initialize input samples
y_sample = zeros(n_tr, 1);    % initialize output samples
var_fctr = [15,2,8,0,10];     % values for qualitative variable (level 1 to level 5)
assert(n_lv == length(var_fctr), 'number of levels is incorrect!');
lhs_iter = 1000;

% Test function (only 1 qualitative and 1 quantitative variables)
obj_handle = @(x) (x(:,2)-5.1/4/pi^2*x(:,1).^2+5/pi*x(:,1)-6).^2+10*(1-1/8/pi)*cos(x(:,1))+10;
% Preparing samples for all levels
for i = 1:n_lv
    A = lhsdesign(n_tr_lv, 1, 'iterations', lhs_iter);
    lhs_temp = A.*repmat(ub-lb, n_tr_lv, 1)+repmat(lb, n_tr_lv, 1);
    X_sample(((i-1)*n_tr_lv+1):(i*n_tr_lv),1:(ind_qual-1)) = lhs_temp;
    X_sample(((i-1)*n_tr_lv+1):(i*n_tr_lv),ind_qual) = i*ones(n_tr_lv,1);
    y_sample(((i-1)*n_tr_lv+1):(i*n_tr_lv),1) = obj_handle([lhs_temp(:,1),repmat(var_fctr(i),n_tr_lv,1)]);
end

acf = 'ei'; % type of acquisition function; options: ei, lcb, ub
num_iter = 30; % no. of BO iterations to be performed

model_options.ind_qual = ind_qual;  % LVGP model fitting requires index of qualitative vars
model_options.dim_z = 2; % dimension of latent space
custom_points = 0;

%% BO
result = bayesian_optimizer(obj_handle, var_fctr, X_sample, y_sample, X_range_continuous, acf, num_iter, custom_points, model_options);

figure
plot(result.Y_min_history,'k-o');
xlabel('Iteration'); ylabel('Best Function Value y^*')

    
