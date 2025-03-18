function bias_variance_decomposition
    % This script demonstrates the bias-variance trade-off by:
    %   1. Generating M independent training sets from a noisy sine function.
    %   2. Fitting a polynomial regression model (fixed degree) to each training set.
    %   3. Evaluating the predictions on a fine grid.
    %   4. Computing the bias^2 and variance at each grid point.
    %
    % The true function is f(x) = sin(2*pi*x). Noise is added to simulate real data.
    
    % Settings
    M = 100;             % Number of training sets (experiments)
    N = 30;              % Number of points per training set
    degree = 3;          % Polynomial degree (fixed for all experiments)
    noise_std = 0.2;     % Standard deviation of noise
    
    % True function
    true_function = @(x) sin(2*pi*x);
    
    % Evaluation grid
    x_fine = linspace(-1, 1, 200)';
    y_true = true_function(x_fine);
    
    % Preallocate matrix to store predictions from each experiment
    predictions = zeros(length(x_fine), M);
    
    % Loop over experiments (each with a new training set)
    for m = 1:M
        % Generate training data: x uniformly in [-1,1] and add noise
        x_train = -1 + 2 * rand(N,1);
        y_train = true_function(x_train) + noise_std * randn(N,1);
        
        % Build the design matrix for the current training set
        X_train = polynomialDesignMatrix(x_train, degree);
        % Compute polynomial coefficients using the normal equation
        w = (X_train' * X_train) \ (X_train' * y_train);
        
        % Evaluate the fitted model on the fine grid
        X_fine = polynomialDesignMatrix(x_fine, degree);
        predictions(:, m) = X_fine * w;
    end
    
    % Compute the average prediction over all experiments
    avg_prediction = mean(predictions, 2);
    
    % Compute the squared bias: (average prediction - true function)^2
    bias_sq = (avg_prediction - y_true).^2;
    
    % Compute the variance: variance of the predictions at each grid point
    variance = var(predictions, 0, 2);  % 0 -> normalization by (M-1)
    
    % Assume known noise variance (for reference)
    noise_variance = noise_std^2;
    
    % Total expected error (per grid point): bias^2 + variance + noise variance
    total_error = bias_sq + variance + noise_variance;
    
    %% Plot the results
    figure('Position',[100 100 900 600]);
    
    % Plot the true function, average prediction, and a few sample fits
    subplot(2,1,1);
    plot(x_fine, y_true, 'k--', 'LineWidth',2, 'DisplayName','True Function'); hold on;
    plot(x_fine, avg_prediction, 'r-', 'LineWidth',2, 'DisplayName','Average Prediction');
    % Plot 10 sample predictions to illustrate variability
    for m = 1:10
        plot(x_fine, predictions(:, m), 'b-', 'LineWidth',1, 'HandleVisibility','off');
    end
    xlabel('x'); ylabel('y');
    title('True Function, Average Prediction, and Sample Fits');
    legend('Location','Best');
    grid on;
    
    % Plot the bias^2, variance, and total error as functions of x
    subplot(2,1,2);
    plot(x_fine, bias_sq, 'r-', 'LineWidth',2, 'DisplayName','Bias^2');
    hold on;
    plot(x_fine, variance, 'b-', 'LineWidth',2, 'DisplayName','Variance');
    plot(x_fine, total_error, 'k-', 'LineWidth',2, 'DisplayName','Total Expected Error');
    xlabel('x'); ylabel('Error');
    title('Error Decomposition Across x');
    legend('Location','Best');
    grid on;
    
    % Display average error components over the grid
    fprintf('Average Bias^2: %.4f\n', mean(bias_sq));
    fprintf('Average Variance: %.4f\n', mean(variance));
    fprintf('Average Total Error (including noise): %.4f\n', mean(total_error));

    % save_all_figs_OPTION('results/bias_variance_decomposition','png',1)
end

%% Helper function: Create polynomial design matrix
function X_poly = polynomialDesignMatrix(x, degree)
    % Constructs the design matrix for polynomial regression.
    % Given a column vector x and a polynomial degree, returns a matrix with columns:
    % [1, x, x.^2, ..., x.^degree]
    
    N = length(x);
    X_poly = ones(N, degree+1);
    for d = 1:degree
        X_poly(:, d+1) = x.^d;
    end
end
