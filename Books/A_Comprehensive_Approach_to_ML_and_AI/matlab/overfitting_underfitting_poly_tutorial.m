function overfitting_vs_underfitting_poly
    % Demonstrates overfitting vs. underfitting using polynomial regression.
    %
    % In this example, we:
    %   1. Generate synthetic data from a sine function with added noise.
    %   2. Fit three polynomial models with different degrees:
    %      - Degree 1: Underfitting (model too simple)
    %      - Degree 5: A moderate fit (good balance)
    %      - Degree 15: Overfitting (model captures noise)
    %   3. Plot the data, true function, and the fitted curves.
    %   4. Compute and display the training Mean Squared Error (MSE).

    % For reproducibility
    rng(1);
    
    %% 1. Generate synthetic data
    N = 30;                      % Number of data points
    x = linspace(-1, 1, N)';      % Feature values (column vector)
    noise = 0.2 * randn(N,1);      % Gaussian noise
    y = sin(2*pi*x) + noise;     % Noisy observations from true function

    % Define polynomial degrees to compare:
    % Degree 1 -> Underfitting, Degree 5 -> Good fit, Degree 15 -> Overfitting
    degrees = [1, 5, 15];

    % Create a fine grid for evaluating the fitted curves
    x_fine = linspace(-1, 1, 200)';

    %% 2. Fit polynomial models and plot
    figure('Position',[100 100 1200 400]);

    for i = 1:length(degrees)
        d = degrees(i);
        
        % Build the design matrix for training data
        X = polynomialDesignMatrix(x, d);
        % Compute coefficients using the normal equation: w = (X'X)^(-1)*X'y
        w = (X'*X) \ (X'*y);
        
        % Evaluate the model on the fine grid
        X_fine = polynomialDesignMatrix(x_fine, d);
        y_pred = X_fine * w;
        
        % Compute training predictions and MSE
        y_train_pred = X * w;
        mse_train = mean((y - y_train_pred).^2);
        
        % Plot the results in a subplot
        subplot(1,3,i);
        hold on;
        % Plot the original noisy data
        scatter(x, y, 50, 'b', 'filled', 'DisplayName','Data');
        % Plot the fitted polynomial curve
        plot(x_fine, y_pred, 'r-', 'LineWidth',2, 'DisplayName', sprintf('Degree %d', d));
        % Plot the true underlying function (without noise) for reference
        y_true = sin(2*pi*x_fine);
        plot(x_fine, y_true, 'k--', 'LineWidth',1.5, 'DisplayName','True Function');
        hold off;
        
        title(sprintf('Degree %d (MSE = %.3f)', d, mse_train));
        xlabel('x'); ylabel('y');
        legend('Location','best');
        grid on;
    end

    % save_all_figs_OPTION('results/overfitting_vs_underfitting_poly','png',1)
end

%% Helper function: Constructs a polynomial design matrix
function X_poly = polynomialDesignMatrix(x, degree)
    % Given a column vector x and a polynomial degree, returns the design
    % matrix X_poly with columns [1, x, x.^2, ..., x.^degree].
    N = length(x);
    X_poly = ones(N, degree+1);
    for p = 1:degree
        X_poly(:, p+1) = x.^p;
    end
end
