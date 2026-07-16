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

    case 'rastrigin_6d'                               % TP-7 (5 continuous dims, 4 levels)
        % f(Xq, lv) = sum_i[(x_i - c_i(lv))^2 - 10 cos(2 pi (x_i - c_i(lv))) + 10] + b(lv)
        % sigma     = sqrt(1 + 0.05*mean(Xq.^2, 2)) * mult(lv)
        % Xq is (n x 5); lv is (n x 1) of level indices. Level-indexed lookups are reshaped to a
        % COLUMN (the row-orientation gotcha above applies to b/muls; C(lv,:) is already n x 5).
        C = [-2  1 -1  2  0;
              1 -2  2 -1 -2;
              2  2 -2  0  1;
             -1 -1  0 -2  2];
        bshift = [5, 20, 40, 65];
        muls   = [1.0, 2.0, 1.5, 3.0];
        f   = @(Xq, lv) sum((Xq - C(round(lv(:)),:)).^2 ...
                            - 10*cos(2*pi*(Xq - C(round(lv(:)),:))) + 10, 2) ...
                        + reshape(bshift(round(lv(:))), [], 1);
        sig = @(Xq, lv) sqrt(1 + 0.05*mean(Xq.^2, 2)) .* reshape(muls(round(lv(:))), [], 1);
        lb = -5*ones(1,5); ub = 5*ones(1,5); n_lv = 4;

    case 'griewank_10d'                               % TP-5 (9 continuous dims, 4 levels)
        % f = sum(xs.^2)/4000 - prod(cos(xs_i/sqrt(i))) + 1 over xs = [Xq, v(lv)] (10 dims)
        vals = [1, 2, 3, 4]; muls = [1.5, 1.0, 3.0, 2.0];
        sq = sqrt(1:10);
        f   = @(Xq, lv) sum([Xq, reshape(vals(round(lv(:))),[],1)].^2, 2)/4000 ...
                        - prod(cos([Xq, reshape(vals(round(lv(:))),[],1)] ./ sq), 2) + 1;
        sig = @(Xq, lv) 0.05*sqrt(1 + 0.1*sum(Xq.^2, 2)/9) .* reshape(muls(round(lv(:))),[],1);
        lb = -5*ones(1,9); ub = 5*ones(1,9); n_lv = 4;

    case 'ackley_10d'                                 % TP-6 (9 continuous dims, 4 levels)
        % f = -20 exp(-0.2 sqrt(sum(xs.^2)/10)) - exp(sum(cos(2 pi xs))/10) + 20 + e, xs=[Xq, v(lv)]
        vals = [1, 2, 3, 4]; muls = [2.0, 1.0, 3.5, 1.5];
        f   = @(Xq, lv) -20*exp(-0.2*sqrt(sum([Xq, reshape(vals(round(lv(:))),[],1)].^2, 2)/10)) ...
                        - exp(sum(cos(2*pi*[Xq, reshape(vals(round(lv(:))),[],1)]), 2)/10) ...
                        + 20 + exp(1);
        sig = @(Xq, lv) 0.10*exp(0.20*sqrt(mean(Xq.^2, 2))) .* reshape(muls(round(lv(:))),[],1);
        lb = -5*ones(1,9); ub = 5*ones(1,9); n_lv = 4;

    case 'golinski'                                   % ENG-1 (6 continuous dims, 5 levels)
        % X columns: x1 face-width, x2 module, x4, x5 shaft lengths, x6, x7 diameters; x3 = teeth(lv)
        vals = [17, 19, 21, 25, 28]; muls = [1.2, 1.0, 1.5, 1.8, 1.3];
        f   = @(Xq, lv) 0.7854*Xq(:,1).*Xq(:,2).^2 ...
                          .* (3.3333*reshape(vals(round(lv(:))),[],1).^2 ...
                              + 14.9334*reshape(vals(round(lv(:))),[],1) - 43.0934) ...
                        - 1.508*Xq(:,1).*(Xq(:,5).^2 + Xq(:,6).^2) ...
                        + 7.4777*(Xq(:,5).^3 + Xq(:,6).^3) ...
                        + 0.7854*(Xq(:,3).*Xq(:,5).^2 + Xq(:,4).*Xq(:,6).^2);
        sig = @(Xq, lv) 20*exp(0.40*(Xq(:,1)-2.6)).*(1 + 0.20*(Xq(:,5)-2.9)) ...
                        .* reshape(muls(round(lv(:))),[],1);
        lb = [2.6, 0.7, 7.3, 7.3, 2.9, 5.0]; ub = [3.6, 0.8, 8.3, 8.3, 3.9, 5.5]; n_lv = 5;

    case 'piston'                                     % ENG-2 (6 continuous dims, 4 levels)
        % X columns: M, S, V0, k, P0, Ta; T0 = level value
        vals = [340, 346, 352, 358]; muls = [1.0, 1.3, 1.6, 2.0];
        f   = @(Xq, lv) piston_cycle(Xq, reshape(vals(round(lv(:))),[],1));
        sig = @(Xq, lv) 0.002*(1 + 3*(Xq(:,5)-90000)/20000).*(1 + 0.5*(Xq(:,1)-30)/30) ...
                        .* reshape(muls(round(lv(:))),[],1);
        lb = [30, 0.005, 0.002, 1000,  90000, 290];
        ub = [60, 0.020, 0.010, 5000, 110000, 296]; n_lv = 4;

    case 'otl_circuit'                                % ENG-3 (5 continuous dims, 4 levels)
        % X columns: Rb1, Rb2, Rf, Rc1, Rc2; beta = level value
        vals = [50, 100, 200, 300]; muls = [2.5, 1.5, 1.0, 0.7];
        f   = @(Xq, lv) otl_vout(Xq, reshape(vals(round(lv(:))),[],1));
        sig = @(Xq, lv) 0.01*sqrt(Xq(:,1)/50).*(1 + 0.3*Xq(:,3)) ...
                        .* reshape(muls(round(lv(:))),[],1);
        lb = [50, 25, 0.5, 1.2, 0.25]; ub = [150, 70, 3.0, 2.5, 1.25]; n_lv = 4;

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

function C = piston_cycle(Xq, T0)
% ENG-2 piston cycle time. Xq = [M, S, V0, k, P0, Ta]; T0 column (level value per row).
    M = Xq(:,1); S = Xq(:,2); V0 = Xq(:,3); k = Xq(:,4); P0 = Xq(:,5); Ta = Xq(:,6);
    A = P0.*S + 19.62*M - k.*V0./S;
    V = (S./(2*k)) .* (sqrt(A.^2 + 4*k.*P0.*V0.*Ta./T0) - A);
    C = 2*pi*sqrt(M ./ (k + S.^2.*P0.*V0.*Ta ./ (T0.*V.^2)));
end

function Vout = otl_vout(Xq, beta)
% ENG-3 OTL mid-point voltage. Xq = [Rb1, Rb2, Rf, Rc1, Rc2]; beta column.
    Rb1 = Xq(:,1); Rb2 = Xq(:,2); Rf = Xq(:,3); Rc1 = Xq(:,4); Rc2 = Xq(:,5);
    Vm  = 12*Rb2 ./ (Rb1 + Rb2);
    bR  = beta .* (Rc2 + 9);
    den = bR + Rf;
    Vout = (Vm + 0.74).*bR./den + 11.35*Rf./den + 0.74*Rf.*bR./(den.*Rc1);
end
