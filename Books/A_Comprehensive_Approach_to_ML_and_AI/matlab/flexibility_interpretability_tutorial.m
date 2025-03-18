function flexibility_vs_interpretability
    % This tutorial demonstrates the flexibility vs. interpretability trade-off.
    % We generate synthetic data from a non-linear function and fit two models:
    %   - A linear model (interpretable but less flexible)
    %   - A 5th degree polynomial model (more flexible but less interpretable)
    %
    % Run this script to see the differences in the fitted curves and the model coefficients.
    
    %% 1. Generate synthetic data
    rng(1);                        % For reproducibility
    N = 50;                        % Number of data points
    x = linspace(0, 1, N)';         % Input feature in [0,1]
    true_function = @(x) sin(2*pi*x); % True underlying function (non-linear)
    noise_std = 0.1;               % Standard deviation of Gaussian noise
    y = true_function(x) + noise_std * randn(N, 1);
    
    %% 2. Fit a linear regression model (interpretable, low flexibility)
    degree_linear = 1;             % Linear model (degree 1)
    X_linear = polynomialDesignMatrix(x, degree_linear);
    w_linear = (X_linear' * X_linear) \ (X_linear' * y); % Normal equation
    
    %% 3. Fit a flexible polynomial regression model (degree 5)
    degree_poly = 5;               % Flexible model (5th degree polynomial)
    X_poly = polynomialDesignMatrix(x, degree_poly);
    w_poly = (X_poly' * X_poly) \ (X_poly' * y);
    
    %% 4. Evaluate predictions on a fine grid
    x_fine = linspace(0, 1, 200)';
    
    % Predictions for the linear model
    X_linear_fine = polynomialDesignMatrix(x_fine, degree_linear);
    y_pred_linear = X_linear_fine * w_linear;
    
    % Predictions for the 5th degree model
    X_poly_fine = polynomialDesignMatrix(x_fine, degree_poly);
    y_pred_poly = X_poly_fine * w_poly;
    
    %% 5. Plot the results
    figure('Position',[100 100 1000 500]);
    
    % Plot for the linear (interpretable) model
    subplot(1,2,1);
    scatter(x, y, 50, 'b', 'filled'); hold on;
    plot(x_fine, true_function(x_fine), 'k--', 'LineWidth',2, ...
         'DisplayName','True Function');
    plot(x_fine, y_pred_linear, 'r-', 'LineWidth',2, ...
         'DisplayName','Linear Fit (Degree 1)');
    xlabel('x'); ylabel('y');
    title('Interpretable Model: Linear Regression');
    legend('Location','Best');
    grid on;
    
    % Plot for the flexible (degree 5) model
    subplot(1,2,2);
    scatter(x, y, 50, 'b', 'filled'); hold on;
    plot(x_fine, true_function(x_fine), 'k--', 'LineWidth',2, ...
         'DisplayName','True Function');
    plot(x_fine, y_pred_poly, 'r-', 'LineWidth',2, ...
         'DisplayName','Polynomial Fit (Degree 5)');
    xlabel('x'); ylabel('y');
    title('Flexible Model: 5th Degree Polynomial Regression');
    legend('Location','Best');
    grid on;
    
    %% 6. Display model coefficients
    fprintf('Linear Regression Coefficients (Interpretable):\n');
    disp(w_linear);
    
    fprintf('5th Degree Polynomial Regression Coefficients (Less Interpretable):\n');
    disp(w_poly);

    % save_all_figs_OPTION('results/flexibility_vs_interpretability','png',1)
end

%% Helper function: Constructs a polynomial design matrix
function X_poly = polynomialDesignMatrix(x, degree)
    % Given a column vector x and a polynomial degree, this function returns
    % the design matrix with columns: [1, x, x.^2, ..., x.^degree]
    N = length(x);
    X_poly = ones(N, degree + 1);
    for d = 1:degree
        X_poly(:, d+1) = x.^d;
    end
end

