function eval_problems(name, in_file, out_file)
% Evaluate problems(name)'s f/sigma on the (x1, lv) grid in in_file and save to out_file,
% so verify_problems.py can compare the MATLAB definition against utils/problems.py.
    here = fileparts(mfilename('fullpath'));
    addpath(here);
    L = load(in_file);                 % L.X (n x d), L.lv (n x 1, same length)
    [f, sig, lb, ub, n_lv] = problems(name);
    fv = f(L.X, L.lv);
    sv = sig(L.X, L.lv);
    save(out_file, 'fv', 'sv', 'lb', 'ub', 'n_lv', '-v7');
end
