close all; clear all; clc; 

% Generate synthetic dataset
rng(42);
X = linspace(1, 10, 100)'; % Feature (house size)
Y = 3 * X + 5 + randn(100,1) * 2; % Target (house price)

% Train Linear Regression
alpha = 0.01;
epochs = 1000;
% model = linear_regression_train(X, Y);
model = linear_regression_train_fast(X, Y);

% Predict function
X_test = [ones(size(X,1),1), X]; % Add bias term
Y_pred = X_test * model.w;

% Compute error
mse = mean((Y - Y_pred).^2);
fprintf('Mean Squared Error: %.4f\n', mse);

% Plot dataset
figure; hold on;
scatter(X, Y, 'bo', 'filled'); % Data points
plot(X, Y_pred, 'r', 'LineWidth', 2); % Regression line
xlabel('Feature (X)'); ylabel('Target (Y)');
title('Linear Regression Fit');
legend('Data Points', 'Best-Fit Line');
grid on;
hold off;

% save_all_figs_OPTION('results/linearRegression','png',1)


function model = linear_regression_train_fast(X, Y)
    % Optimized solution using Normal Equation
    % Inputs:
    %   X - Feature matrix (Nx1)
    %   Y - Target values (Nx1)
    % Output:
    %   model - Struct containing trained weights
    
    N = length(Y);
    X = [ones(N, 1), X]; % Add bias term
    w = pinv(X' * X) * X' * Y; % Normal Equation
    
    model.w = w;
end


function model = linear_regression_train(X, Y, alpha, epochs)
    % Inputs:
    %   X - Feature matrix (Nx1)
    %   Y - Target values (Nx1)
    %   alpha - Learning rate
    %   epochs - Number of training iterations
    % Output:
    %   model - Struct containing trained weights
    
    N = length(Y);
    X = [ones(N, 1), X]; % Add bias term
    w = zeros(2, 1);  % Initialize weights (w and b)
    
    for epoch = 1:epochs
        % Compute predictions
        Y_pred = X * w;
        
        % Compute gradients
        gradient = (X' * (Y_pred - Y)) / N;
        
        % Update weights
        w = w - alpha * gradient;
        
        % Compute loss (MSE)
        loss = mean((Y - Y_pred).^2);
        
        if mod(epoch, 100) == 0
            fprintf('Epoch %d, Loss: %.4f\n', epoch, loss);
        end
    end
    
    model.w = w;
end
