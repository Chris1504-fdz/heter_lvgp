function result = bayesian_optimizer(objfunc, var_fctr, X_sample, y_sample, X_range_continuous, acf, num_iter, custom_points, model_options)
% variables:
        % objfunc - objective function to optimize
        % X_Sample, y_sample - available sample data, cols represent varibales and rows represent samples
        % X_range_continuous - range of continuous variables, first row is lowerbound and second row is upperbound
        % acf - acquisition function, recommend to use 'ei', other options are 'pi' and 'lcb'
        % num_iter - maximal number of BO iterations
        % custom_points - points to search, set to 0 if not supplied
        % model_options - LVGP model fitting options. For all available
        % options, check LVGP_fit.m
% -- by Yichi Zhang, Northwestern University, @yichizhang2013@u.northwestern.edu
% -- modified by Akshay Iyer, Rick Tsai

y_min = min(y_sample);
U = zeros(num_iter, 1);
Y_min = zeros(num_iter, 1); Y_min_est = Y_min;
X_min_est = zeros(num_iter, size(X_sample, 2));
X_sampled = X_sample; y_sampled = y_sample;
n_points = 2000*size(X_sample,2);
PlotOrNot = 'Plot'; % If you do not want plots, please change it to 'DontPlot'
for i = 1:num_iter

    % LVPGP model fitting
    model = LVGP_fit(X_sampled, y_sampled, model_options);

    % Generating the plots
    if strcmp(PlotOrNot,'Plot')==1
        % Define plotting range for x
        x_grid = linspace(min(X_sampled(:,1)), max(X_sampled(:,1)), 100)';
        
        % Create figure
        figure('Position', [100, 100, 1200, 400]);  % Wider figure
        tiledlayout(2,3); % 5 subplots, using 3x2 layout
    
        % Loop over category levels (1 to 5)
        for level = 1:5
            % Get subplot position
            nexttile;
        
            % Filter original data by category level
            idx = (X_sampled(:,2) == level);
            x_data = X_sampled(idx, 1);
            y_data = y_sampled(idx);
        
            % Construct prediction input (x_grid with current level)
            X_pred = [x_grid, level * ones(size(x_grid))];
        
            % Predict using the trained GP model
            output = LVGP_predict(X_pred, model, 'MSE_on', true);
            y_pred = output.Y_hat; y_cov = output.MSE;
            y_std = sqrt(abs(diag(y_cov)));
        
            % Plot GP mean prediction
            plot(x_grid, y_pred, 'b-', 'LineWidth', 1.5); hold on;
        
            % Plot 95% confidence interval
            fill([x_grid; flipud(x_grid)], ...
                 [y_pred - 1.96*y_std; flipud(y_pred + 1.96*y_std)], ...
                 [0.8 0.8 1], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
        
            % Plot sampled data points
            plot(x_data, y_data, 'ko', 'MarkerFaceColor', 'k');
        
            title(['Category Level ', num2str(level)]);
            xlabel('x');
            ylabel('y');
            grid on;
        end
        nexttile(1); ylim([-10,300]); nexttile(2); ylim([-10,250]); nexttile(3); ylim([-10,100]); 
        nexttile(4); ylim([-10,350]); nexttile(5); ylim([-10,100]);
    end

    % find min and the next sampling point
    [x_next, U_min_est, x_min_est, y_min_est] = find_next(model, X_range_continuous, acf, n_points, custom_points, X_sampled, y_sampled);
    x_next_eval = [x_next(1),var_fctr(int32(x_next(2)))];
    y_next = objfunc(x_next_eval);
    

    % Plot x_next
    if strcmp(PlotOrNot,'Plot')==1
        nexttile(int32(x_next(2)));
        plot(x_next(1), y_next, 'r*');
        sgtitle(sprintf('Bayesian Optimization using LVGP - Iteration %d', i));
        saveas(gcf, sprintf('LVGP_BO_plots/figure_iter_%03d.png', i));  % or use exportgraphics(gcf, filename) for better quality
        close(gcf);
    end

    y_min = min(y_next, y_min);

    X_sampled = [X_sampled; x_next]; y_sampled = [y_sampled; y_next];

    Y_min(i) = y_min; Y_min_est(i) = y_min_est;
    X_min_est(i,:) = x_min_est; 
    U(i) =  - U_min_est;

    fprintf('Iteration %u completed. ', i) 
    fprintf('Current minima is %f.\n', y_min);

end

result.Y_optimum = y_min;
result.Y_min_history = Y_min;
result.X_sampled = X_sampled;
result.Y_sampled = y_sampled;
result.acf_val = U;
result.final_model = model;




