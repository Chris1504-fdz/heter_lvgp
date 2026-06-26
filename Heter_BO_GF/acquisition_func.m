function U_negate = acquisition_func(model, x_pred, y_min, acf, y_pred, s, bo_options)
% Acquisition functions for heteroscedastic LVGP BO.
% Returns a value to be minimized by fmincon; EI/PI/HAEI are negated.
%
% acf options:
%   'ei'   : expected improvement
%   'lcb'  : lower confidence bound using epistemic std
%   'pi'   : probability of improvement
%   'haei' : heteroscedastic augmented expected improvement
%   'rahbo': risk-averse heteroscedastic BO acquisition
%
% HAEI follows Griffiths et al. (2022):
%   HAEI(x) = EI(x) * (1 - gamma*sqrt(r(x))/sqrt(var_epi(x)+gamma^2*r(x)))
% where r(x) is predicted aleatoric variance and var_epi(x) is GP epistemic variance.

if nargin < 3
    error('not enough inputs for acquisition function');
end
if nargin < 4 || isempty(acf)
    acf = 'ei';
end
if nargin < 7 || isempty(bo_options)
    bo_options = struct();
end
if ~isfield(bo_options, 'gamma')
    bo_options.gamma = 1;
end
if ~isfield(bo_options, 'alpha')
    bo_options.alpha = 1;
end
if ~isfield(bo_options, 'beta')
    bo_options.beta = 2;
end

% If predictions are not supplied, compute them with the heteroscedastic LVGP.
if nargin < 5 || isempty(y_pred) || isempty(s)
    if isfield(model, 'data') && isfield(model.data, 'Y_var')
        y_var_train = model.data.Y_var;
    elseif isfield(model, 'y_var_sample')
        y_var_train = model.y_var_sample;
    else
        % Most LVGP_fit_noise versions accept the training variance as the
        % second argument to LVGP_predict_noise. If your implementation stores
        % it internally, replace [] by the correct stored field if needed.
        y_var_train = [];
    end
    y = LVGP_predict_noise(x_pred, y_var_train, model, 'MSE_on', true);
    y_pred = y.Y_hat(:);
    s = sqrt(max(abs(diag(y.MSE)), 1e-12)); % epistemic std
else
    y_pred = y_pred(:);
    s = s(:);
end

s = max(s, 1e-12);
b = (y_min - y_pred)./s;
EI = (y_min - y_pred).*normcdf(b) + s.*normpdf(b);

if strcmpi(acf, 'ei')
    U_negate = -EI;

elseif strcmpi(acf, 'lcb')
    LCB = y_pred - 2*s;
    U_negate = LCB;

elseif strcmpi(acf, 'pi')
    b_pi = (y_min - y_pred - 0.01)./s;
    PI = normcdf(b_pi);
    U_negate = -PI;

elseif strcmpi(acf, 'haei')
    gamma = bo_options.gamma;
    r = predict_aleatoric_variance(x_pred, model); % aleatoric variance r(x)
    var_epi = s.^2;
    scale = 1 - (gamma .* sqrt(r)) ./ sqrt(var_epi + gamma^2 .* r);
    scale = max(scale, 0); % numerical safety; HAEI should be non-negative
    HAEI = EI .* scale;
    U_negate = -HAEI;

elseif strcmpi(acf, 'rahbo')
    % Risk-averse Heteroscedastic BO (RAHBO) acquisition.
    %
    % The paper defines the maximization objective:
    %   MV(x) = f(x) - alpha*rho^2(x)
    % and selects:
    %   argmax_x ucb_f(x) - alpha*lcb_var(x).
    %
    % This code is written for minimization, so we use the minimization
    % analogue:
    %   argmin_x lcb_f(x) + alpha*lcb_var(x).
    %
    % Since the current aleatoric model is a polynomial regression model
    % that provides only a point prediction for r(x), we use:
    %   lcb_var(x) ~= r(x).
    % If you later replace it with a GP variance model, change r below to
    %   lcb_var = mu_var - beta_var*sigma_var.
    alpha = bo_options.alpha;
    beta = bo_options.beta;
    r = predict_aleatoric_variance(x_pred, model); % aleatoric variance rho^2(x)

    lcb_f = y_pred - beta .* s;       % optimistic for minimization
    lcb_var = r;                      % current approximation
    RAHBO = lcb_f + alpha .* lcb_var; % minimize mean + risk penalty

    U_negate = RAHBO;
elseif strcmp(acf,'anpei') == 1

    % ANPEI from Griffiths et al.:
    % ANPEI(x) = beta*EI(x) - (1-beta)*sqrt(r(x))
    %
    % Since fmincon minimizes, return negative ANPEI.

    if ~isfield(bo_options, 'beta_anpei')
        beta_anpei = 0.5;
    else
        beta_anpei = bo_options.beta_anpei;
    end

    % Predicted aleatoric variance
    ale_var = predict_aleatoric_variance(x_pred, model);
    ale_std = sqrt(max(ale_var, 1e-12));


    % ANPEI acquisition, maximized in the paper
    ANPEI = beta_anpei .* EI - (1 - beta_anpei) .* ale_std;

    % fmincon minimizes, so negate it
    U_negate = -ANPEI;
elseif strcmpi(acf, 'gf') % for global fitting
    U_negate = -s.^2;
elseif strcmpi(acf, 'gf_new') % for global fitting
    r = predict_aleatoric_variance(x_pred, model); % aleatoric variance r(x)
    U_negate = -(s.^2-r);
else
    error('Unknown acquisition function: %s', acf);
end
end

function var_pred = predict_aleatoric_variance(Xnew, model)
if ~isfield(model, 'aleatoric_poly')
    error('model.aleatoric_poly is required for HAEI. Fit it after LVGP_fit_noise.');
end
poly = model.aleatoric_poly;
Wnew = zeros(size(Xnew,1), numel(poly.cont_idx) + size(poly.Z_latent,2));
for ii = 1:size(Xnew,1)
    c = Xnew(ii, poly.ind_qual);
    Wnew(ii,:) = [Xnew(ii, poly.cont_idx), poly.Z_latent(c,:)];
end
Wn = (Wnew - poly.mu_W) ./ poly.std_W;
Phi = build_poly_features(Wn, poly.degree);
log_sigma = Phi * poly.theta;
var_pred = exp(2*log_sigma);
var_pred = max(var_pred, 1e-12);
end

function Phi = build_poly_features(Wn, degree)
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
