function run_all_parallel(num_iter, n_workers, results_dir)
% Parallel version of run_all for a machine where the license WORKS (e.g. your
% PC). ONE MATLAB session = ONE license checkout; the local parpool workers run
% under your license (Parallel Computing Toolbox required) -- no extra checkouts.
% Resumable: cells that already have a .mat are skipped.
%
% Usage (from the study/ folder):
%   run_all_parallel               % 4 workers, num_iter=30
%   run_all_parallel(30, 4)        % 30 iters, 4 workers
%   run_all_parallel(30, 6)        % more workers if you have the RAM (~1.5 GB each)
%
% NOTE: 16 GB RAM -> ~4 workers is safe. Do NOT open multiple MATLAB windows;
% this single session + parpool is the safe way to use multiple cores.

if nargin < 1 || isempty(num_iter),   num_iter = 30; end
if nargin < 2 || isempty(n_workers),  n_workers = 4;  end
here = fileparts(mfilename('fullpath'));
if nargin < 3 || isempty(results_dir), results_dir = fullfile(here, 'results'); end
addpath(here);
addpath(fullfile(here, '..', 'LVGP_Matlab_codes'));
addpath(fullfile(here, '..', 'Heter_BO_GF'));

% --- build the list of PENDING jobs ---
configs = { ...
  'lcb',  NaN; 'pi',   NaN; 'ei',   NaN; ...
  'haei', 0.5; 'haei', 1.0; 'haei', 5.0; ...
  'anpei',0.2; 'anpei',0.5; 'anpei',0.8; ...
  'rahbo',0.5; 'rahbo',1.0; 'rahbo',5.0 };
n_reps = [3 5 10];
seeds  = 1:30;

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
            acfs{end+1} = acf;        %#ok<AGROW>
            params(end+1) = param;    %#ok<AGROW>
            nreps(end+1) = nr;        %#ok<AGROW>
            sds(end+1) = s;           %#ok<AGROW>
            outs{end+1} = out;        %#ok<AGROW>
        end
    end
end
n = numel(outs);
fprintf('%d pending jobs across %d workers (num_iter=%d)\n', n, n_workers, num_iter);
if n == 0, fprintf('nothing pending.\n'); return; end

% --- start a local pool (idempotent) ---
pool = gcp('nocreate');
if isempty(pool)
    parpool('local', n_workers);
elseif pool.NumWorkers ~= n_workers
    delete(pool); parpool('local', n_workers);
end

% --- run in parallel ---
t0 = tic;
parfor j = 1:n
    maxNumCompThreads(1);   % CRITICAL: tiny LVGP matrices -> 1 thread is ~10x faster
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
