function run_chunk(chunk_file, num_iter)
% Persistent worker: one MATLAB session processes an entire list of BO jobs in a
% loop, so the license is checked out ONCE (at session start) instead of once
% per run. Each line of chunk_file is: acf,param,n_rep,seed,outpath
maxNumCompThreads(1);   % CRITICAL: tiny LVGP matrices -> 1 thread is ~10x faster
here = fileparts(mfilename('fullpath'));
addpath(here);
addpath(fullfile(here, '..', 'LVGP_Matlab_codes'));
addpath(fullfile(here, '..', 'Heter_BO_GF'));

fid = fopen(chunk_file, 'r');
lines = {};
tline = fgetl(fid);
while ischar(tline)
    if ~isempty(strtrim(tline)), lines{end+1} = tline; end %#ok<AGROW>
    tline = fgetl(fid);
end
fclose(fid);
fprintf('run_chunk: %d jobs from %s\n', numel(lines), chunk_file);

for i = 1:numel(lines)
    parts = strsplit(lines{i}, ',');
    acf     = strtrim(parts{1});
    param   = str2double(parts{2});      % NaN for ei/lcb/pi
    n_rep   = str2double(parts{3});
    seed    = str2double(parts{4});
    outpath = strtrim(parts{5});
    if exist(outpath, 'file')
        fprintf('SKIP %s\n', outpath); continue;
    end
    try
        study_driver(acf, param, n_rep, seed, num_iter, outpath);
        fprintf('OK %s\n', outpath);
    catch ME
        fprintf('CHUNKFAIL %s : %s\n', outpath, ME.message);
    end
end
fprintf('run_chunk DONE: %s\n', chunk_file);
end
