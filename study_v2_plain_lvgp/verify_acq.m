function verify_acq(out_file)
% Exercise the REAL BO_standard_LVGP/acquisition_func.m on a grid of (mu, s, y_min)
% and dump (mu, s, y_min, EI, LCB, PI) so an independent Python script can check it
% against closed-form + Monte-Carlo definitions. acquisition_func, when called with
% y_pred and s supplied (nargin==6), uses them directly and never touches the GP model,
% so we can probe it with a dummy model.
    here = fileparts(mfilename('fullpath'));
    addpath(fullfile(here, 'BO_standard_LVGP'));

    mus   = [-5 -3 -1 -0.2 0 0.3 0.5 0.8 1 2 5 10];   % y_pred (posterior mean)
    ss    = [0.05 0.1 0.3 0.5 1 2 5];                  % posterior std
    ymins = [-1 0.5 3];                                % incumbents
    dummy = struct();                                  % model is unused when mu,s passed

    rows = [];
    for a = 1:numel(ymins)
      ymin = ymins(a);
      for i = 1:numel(mus)
        for j = 1:numel(ss)
          yp = mus(i); s = ss(j);
          ei  = -acquisition_func(dummy, [], ymin, 'ei',  yp, s);   % U_negate = -EI
          lcb =  acquisition_func(dummy, [], ymin, 'lcb', yp, s);   % U_negate = LCB
          pid = -acquisition_func(dummy, [], ymin, 'pi',  yp, s);   % U_negate = -PI
          rows = [rows; yp, s, ymin, ei, lcb, pid];
        end
      end
    end
    save(out_file, 'rows', '-v7');
    fprintf('verify_acq: wrote %d rows to %s\n', size(rows,1), out_file);
end
