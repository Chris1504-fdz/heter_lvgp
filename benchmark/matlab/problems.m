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

    case 'sixhump_camel'                              % TP-2 (1-D, 4 levels)
        vals = [0.2, 0.4, 0.7, 1.0]; muls = [2.0, 3.5, 1.5, 5.0];
        f   = @(x1, lv) (4 - 2.1*x1.^2 + x1.^4/3).*x1.^2 + x1.*reshape(vals(lv),size(x1)) ...
                        + (-4 + 4*reshape(vals(lv),size(x1)).^2).*reshape(vals(lv),size(x1)).^2;
        sig = @(x1, lv) 0.05 .* exp((0.4.*x1).^2) .* reshape(muls(lv),size(x1));
        lb = -2; ub = 2; n_lv = 4;

    case 'griewank_2d'                                % TP-3 (1-D, 4 levels)
        vals = [0.0, 0.5, 1.0, 1.5]; muls = [2.0, 1.0, 3.5, 1.5];
        f   = @(x1, lv) (x1.^2 + reshape(vals(lv),size(x1)).^2)/4000 ...
                        - cos(x1).*cos(reshape(vals(lv),size(x1))/sqrt(2)) + 1;
        sig = @(x1, lv) 0.04 .* (1 + 0.08*x1.^2) .* reshape(muls(lv),size(x1));
        lb = -5; ub = 5; n_lv = 4;

    case 'ackley_2d'                                  % TP-4 (1-D, 4 levels)
        vals = [0.0, 0.5, 1.5, 2.5]; muls = [1.0, 2.0, 1.5, 3.0]; a = 20; b = 0.2; c = 2*pi;
        f   = @(x1, lv) -a*exp(-b*sqrt((x1.^2 + reshape(vals(lv),size(x1)).^2)/2)) ...
                        - exp((cos(c*x1) + cos(c*reshape(vals(lv),size(x1))))/2) + a + exp(1);
        sig = @(x1, lv) 0.10 .* (1 + 0.15*x1.^2) .* reshape(muls(lv),size(x1));
        lb = -3; ub = 3; n_lv = 4;

    otherwise
        error('problems:undefined', ...
              'function "%s" is not defined in matlab/problems.m (mirror it from utils/problems.py)', name);
end
end
