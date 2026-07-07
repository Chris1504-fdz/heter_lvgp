function [f, sig, lb, ub, n_lv] = problems(name)
% Function registry for the MATLAB (LVGP) engine.
% Returns handles f(x1,level) -> noise-free objective, sig(x1,level) -> noise std, plus the
% continuous domain [lb,ub] and the number of categorical levels n_lv. `level` is 1-based.
%
% MUST MIRROR utils/problems.py -- verify_problems.py checks the two agree on a grid of points.
% Add the other 8 functions here as their equations are supplied.
switch name
    case 'branin_hetero'
        var_fctr   = [15, 2, 8, 0, 10];               % Branin x2 value per level
        noise_muls = [1.00, 0.70, 0.90, 0.50, 1.20]*10;
        % NOTE: `var_fctr(lv)` keeps var_fctr's (row) orientation, so reshape any level-indexed
        % lookup to size(x1) before combining with x1 (else row-minus-column broadcasts to a matrix).
        f   = @(x1, lv) (reshape(var_fctr(lv), size(x1)) - 5.1/(4*pi^2)*x1.^2 + 5/pi*x1 - 6).^2 ...
                        + 10*(1 - 1/(8*pi))*cos(x1) + 10;
        sig = @(x1, lv) 0.135 .* exp((0.15 .* x1).^2) .* reshape(noise_muls(lv), size(x1));
        lb = -5; ub = 10; n_lv = 5;

    % --- 8 stubs: mirror each from utils/problems.py, then delete this comment ---
    % case 'fn2'
    %     f = @(x1,lv) ...; sig = @(x1,lv) ...; lb = ...; ub = ...; n_lv = ...;

    otherwise
        error('problems:undefined', ...
              'function "%s" is not defined in matlab/problems.m (mirror it from utils/problems.py)', name);
end
end
