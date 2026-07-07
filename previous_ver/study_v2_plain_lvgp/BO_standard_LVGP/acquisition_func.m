function U_negate = acquisition_func(model, x_pred, y_min, acf, y_pred, s)
%  calculate expected improvement for selecting next sampling point
%  X_pred - (n_samples, n_features), numpy array
%  y_pred - (n_samples, 1), numpy array
%  s - (n_samples,), numpy array
%  y_min - scalar
	
if nargin < 3
    
    error('not enough inputs for aquisition function');
    
elseif nargin == 3
    
    acf = 'ei';
    
elseif nargin == 4

    y = LVGP_predict(x_pred, model, 'MSE_on', true);
    y_pred = y.Y_hat; y_cov = y.MSE;
    s = sqrt(abs(diag(y_cov)));

end


if strcmp(acf, 'ei')
        b = (y_min - y_pred)./s;
        EI = (y_min - y_pred).*normcdf(b) + s.*normpdf(b);
        U_negate =  - EI;

elseif strcmp(acf, 'lcb')
        LCB = y_pred - 2*s;
        U_negate = LCB;

elseif strcmp(acf, 'pi')
        b = (y_min - y_pred - 0.01)./s;
        PI = normcdf(b);
        U_negate = -PI;
end