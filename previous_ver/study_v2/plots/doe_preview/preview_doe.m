function preview_doe(out_file)
% Generate ONLY the initial DOE (no BO) for all seeds, using the same RNG stream
% and SHARED-LHS logic as study_driver.m, so the preview matches the real runs.
% The DOE x1 locations are drawn once per seed and shared across all categories,
% so they no longer depend on n_rep. Saves rows = [seed, level, x1].

    lb = -5; ub = 10; n_lv = 5; n_tr_lv = 2;

    rows = [];
    for seed = 1:30
        rng(seed);
        % --- identical to study_driver.m DOE block (maximin, 1/6 inset) ---
        edge_buf = 1/6;
        lo = lb + edge_buf*(ub - lb);                % -2.5 (center of lower third)
        hi = ub - edge_buf*(ub - lb);                %  7.5 (center of upper third)
        A = lhsdesign(n_tr_lv, 1, 'iterations', 1000);
        lhs_shared = A.*(hi - lo) + lo;
        for i = 1:n_lv
            for j = 1:n_tr_lv
                rows(end+1,:) = [seed, i, lhs_shared(j)]; %#ok<AGROW>
            end
        end
    end
    save(out_file, 'rows', '-v7');
    fprintf('DONE preview_doe: %d rows\n', size(rows,1));
end
