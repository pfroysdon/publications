function bias_variance_tutorial
    % Demonstrates the bias-variance trade-off using polynomial regression.
    %
    % We:
    %   1) Generate noisy data from a known function.
    %   2) Fit polynomial regressions of different degrees.
    %   3) Compare training vs test errors.
    %   4) Observe how increasing model complexity affects bias and variance.

    %% 1. Generate synthetic data
    rng(0); % For reproducibility
    
    N = 50;             % Total number of data points
    x = linspace(-1,1,N)';   % Features
    true_function = @(x) sin(2*pi*x);  % Underlying function
    noise_variance = 0.2;    % Noise level
    y = true_function(x) + noise_variance*randn(N,1);  % Observed data

    % Split into train (70%) and test (30%)
    Ntrain = round(0.7*N);
    x_train = x(1:Ntrain);
    y_train = y(1:Ntrain);
    x_test  = x(Ntrain+1:end);
    y_test  = y(Ntrain+1:end);

    %% 2. Fit polynomial models of different degrees
    max_degree = 10;  % Maximum polynomial degree to try
    train_errors = zeros(max_degree,1);
    test_errors  = zeros(max_degree,1);

    for d = 1:max_degree
        % Build polynomial design matrix for training data
        Xtrain = polynomialDesignMatrix(x_train, d);
        % Solve for parameters (normal equation: w = (X'X)^(-1) X'y )
        w = (Xtrain'*Xtrain)\(Xtrain'*y_train);

        % Predictions on training set
        y_pred_train = Xtrain * w;
        train_errors(d) = mean((y_train - y_pred_train).^2);

        % Build polynomial design matrix for test data
        Xtest = polynomialDesignMatrix(x_test, d);
        % Predictions on test set
        y_pred_test = Xtest * w;
        test_errors(d) = mean((y_test - y_pred_test).^2);
    end

    %% 3. Plot the training and test errors
    figure('Position',[100 100 900 400]);

    % Subplot 1: Training & Test Errors vs. Polynomial Degree
    subplot(1,2,1);
    plot(1:max_degree, train_errors, 'o--', 'LineWidth',1.5, 'MarkerSize',8);
    hold on;
    plot(1:max_degree, test_errors, 'o--', 'LineWidth',1.5, 'MarkerSize',8);
    xlabel('Polynomial Degree');
    ylabel('Mean Squared Error (MSE)');
    legend('Training Error','Test Error','Location','best');
    title('Bias-Variance Trade-off');
    grid on;

    % Subplot 2: Visual comparison for a selected degree
    subplot(1,2,2);
    d_select = 5;  % Example: pick a polynomial degree to visualize
    Xtrain_select = polynomialDesignMatrix(x_train, d_select);
    w_select = (Xtrain_select'*Xtrain_select)\(Xtrain_select'*y_train);

    % Plot training data
    plot(x_train, y_train, 'bo', 'DisplayName','Training Data'); hold on;

    % Plot true function (smooth curve)
    x_fine = linspace(-1,1,200)';
    y_true_fine = true_function(x_fine);
    plot(x_fine, y_true_fine, 'k-', 'LineWidth',1.5, 'DisplayName','True Function');

    % Plot polynomial model's prediction
    X_fine_select = polynomialDesignMatrix(x_fine, d_select);
    y_pred_fine = X_fine_select * w_select;
    plot(x_fine, y_pred_fine, 'r-', 'LineWidth',1.5, 'DisplayName',['Degree ' num2str(d_select)]);

    legend('Location','best');
    title(['Polynomial Fit (Degree = ' num2str(d_select) ')']);
    xlabel('x'); ylabel('y');
    grid on;

end

%% Helper function: build polynomial design matrix
function Xpoly = polynomialDesignMatrix(x, degree)
    % Given a column vector x of size Nx1 and a polynomial degree,
    % returns the Nx(degree+1) design matrix:
    % [1, x, x^2, ..., x^degree]
    N = length(x);
    Xpoly = zeros(N, degree+1);
    for d = 0:degree
        Xpoly(:, d+1) = x.^d;
    end
end
