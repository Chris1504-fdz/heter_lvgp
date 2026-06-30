function run_all_parallel(num_iter, n_workers, results_dir)
% Single-MATLAB-session sweep for the STANDARD (homoscedastic) LVGP study.
%
% ONE MATLAB session = ONE license checkout; a local parpool runs the cells in parallel under
% that single license (Parallel Computing Toolbox) -- NO per-cell license checkouts. This is the
% smooth way to use the online (MHLM) license: it avoids the "12 simultaneous checkouts hang"
% failure mode of the per-cell run_sweep.py launcher. Mirrors ../study_v2/run_all_parallel.m.
% Resumable: cells that already have a .mat are skipped.
%
% Only the 3 noise-blind acquisitions (ei/lcb/pi) are swept (plain LVGP is noise-unaware).
%
% Usage (ONE matlab, headless or interactive):
%   matlab -batch "run_all_parallel"            % 4 workers, num_iter=30
%   matlab -batch "run_all_parallel(30, 6)"     % 6 workers
%   run_all_parallel(30, 4)                      % from an interactive session

if nargin < 1 || isempty(num_iter),   num_iter = 30; end
if nargin < 2 || isempty(n_workers),  n_workers = 4;  end
here = fileparts(mfilename('fullpath'));
if nargin < 3 || isempty(results_dir), results_dir = fullfile(here, 'results'); end
addpath(here);                              % study_driver + acf_tag (study_driver adds BO_standard_LVGP)

% --- build the list of PENDING jobs: ei/lcb/pi x n_rep{3,5,10} x seed{1..30} ---
configs = { 'lcb', NaN; 'pi', NaN; 'ei', NaN };
n_reps  = [3 5 10];
seeds   = 1:30;

acfs = {}; params = []; nreps = []; sds = []; outs = {};
for c = 1:size(configs,1)
    acf = configs{c,1}; param = configs{c,2};
    tag = acf_tag(acf, param);
    for nr = n_reps
        outdir = fullfile(results_dir, tag, sprintf('nrep%02d', nr));
        if ~exist(outdir, 'dir'), mkdir(outdir); end
        for s = seeds
            out = fullfile(outdir, sprintf('seed%02d.mat', s));
            if exist(out, 'file'), continue; end
            acfs{end+1} = acf;     %#ok<AGROW>
            params(end+1) = param; %#ok<AGROW>
            nreps(end+1) = nr;     %#ok<AGROW>
            sds(end+1) = s;        %#ok<AGROW>
            outs{end+1} = out;     %#ok<AGROW>
        end
    end
end
n = numel(outs);
fprintf('%d pending jobs across %d workers (num_iter=%d)\n', n, n_workers, num_iter);
if n == 0, fprintf('nothing pending.\n'); return; end

% --- one local pool (idempotent) ---
pool = gcp('nocreate');
if isempty(pool)
    parpool('local', n_workers);
elseif pool.NumWorkers ~= n_workers
    delete(pool); parpool('local', n_workers);
end

% --- run in parallel ---
t0 = tic;
parfor j = 1:n
    maxNumCompThreads(1);   % tiny LVGP matrices -> 1 thread is fastest; avoids oversubscription
    try
        study_driver(acfs{j}, params(j), double(nreps(j)), double(sds(j)), num_iter, outs{j});
        fprintf('OK   %s\n', outs{j});
    catch ME
        warning('FAIL %s : %s', outs{j}, ME.message);
    end
end
fprintf('\nFINISHED %d jobs in %.1f min.\n', n, toc(t0)/60);
end

function t = acf_tag(acf, param)
if isnan(param)
    t = acf;
else
    switch acf
        case 'haei',  letter = 'g';
        case 'rahbo', letter = 'a';
        case 'anpei', letter = 'b';
        otherwise,    letter = 'p';
    end
    t = sprintf('%s_%s%g', acf, letter, param);
end
end
