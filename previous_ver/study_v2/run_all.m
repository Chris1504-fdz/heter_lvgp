function run_all(num_iter, results_dir)
% Pure-MATLAB sequential driver for the whole study.
% Runs every grid cell one after another in a SINGLE MATLAB session, so the
% license is checked out ONCE (at startup) and held — no per-run launches.
% Resumable: any cell that already has a .mat is skipped, so it picks up
% wherever previous runs (e.g. on the cluster) left off.
%
% Usage (from the study/ folder, in MATLAB):
%   run_all                              % num_iter=30, results in ./results
%   run_all(30)                          % same
%   run_all(30, 'C:\path\to\results')    % point at an existing results folder
%
% Whatever .mat already exist (in the nested layout
% <results_dir>/<acf>/nrep<NN>/seed<NN>.mat) are SKIPPED, so just drop your
% existing results there and only the missing cells run.
%
% Requires sibling folders ../LVGP_Matlab_codes and ../Heter_BO_GF and the
% study_driver.m in this folder. Works on Windows/Mac/Linux MATLAB.

if nargin < 1 || isempty(num_iter), num_iter = 30; end
maxNumCompThreads(1);   % CRITICAL: tiny LVGP matrices -> 1 thread is ~10x faster
here = fileparts(mfilename('fullpath'));
if nargin < 2 || isempty(results_dir), results_dir = fullfile(here, 'results'); end
addpath(here);
addpath(fullfile(here, '..', 'LVGP_Matlab_codes'));
addpath(fullfile(here, '..', 'Heter_BO_GF'));
fprintf('results folder: %s\n', results_dir);

% --- the 12 acquisition configs (param = NaN means no knob) ---
configs = { ...
  'lcb',  NaN; 'pi',   NaN; 'ei',   NaN; ...
  'haei', 0.5; 'haei', 1.0; 'haei', 5.0; ...
  'anpei',0.2; 'anpei',0.5; 'anpei',0.8; ...
  'rahbo',0.5; 'rahbo',1.0; 'rahbo',5.0 };
n_reps = [3 5 10];
seeds  = 1:30;

total = size(configs,1) * numel(n_reps) * numel(seeds);
idx = 0; ran = 0; failed = 0; t0 = tic;

for c = 1:size(configs,1)
    acf = configs{c,1}; param = configs{c,2};
    tag = acf_tag(acf, param);
    for nr = n_reps
        outdir = fullfile(results_dir, tag, sprintf('nrep%02d', nr));
        if ~exist(outdir, 'dir'), mkdir(outdir); end
        for s = seeds
            idx = idx + 1;
            out = fullfile(outdir, sprintf('seed%02d.mat', s));
            if exist(out, 'file'), continue; end           % resume
            fprintf('[%d/%d] %s nrep%d seed%d ...', idx, total, tag, nr, s);
            try
                study_driver(acf, param, nr, double(s), num_iter, out);
                ran = ran + 1; fprintf(' done\n');
            catch ME
                failed = failed + 1; fprintf(' FAIL: %s\n', ME.message);
            end
        end
    end
end
fprintf('\nFINISHED: ran %d new cells, %d failed, %.1f min elapsed.\n', ...
        ran, failed, toc(t0)/60);
end

function t = acf_tag(acf, param)
% folder name matching the study layout: ei / lcb / pi / haei_g0.5 / rahbo_a1 / anpei_b0.2
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