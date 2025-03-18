close all; clear all; clc; 

% % Generate synthetic dataset
% rng(42);
% X = [randn(50,2) + 2; randn(50,2) - 2]; % Two Gaussian clusters
% Y = [ones(50,1); zeros(50,1)];
% 
% % Train Logistic Regression
% alpha = 0.1;
% epochs = 1000;
% model = logistic_regression_train_fast(X, Y, alpha, epochs);
% 
% % Predict function
% Y_pred = round(1 ./ (1 + exp(-[ones(size(X,1),1), X] * model.w)));
% 
% % Compute accuracy
% accuracy = mean(Y_pred == Y) * 100;
% fprintf('Logistic Regression Accuracy: %.2f%%\n', accuracy);

% Generate synthetic dataset
rng(42); % Ensure reproducibility
num_samples = 100;

% Class 1 (Blue)
X1 = randn(num_samples/2, 2) + 2;  % Centered at (2,2)
Y1 = ones(num_samples/2, 1);       % Class 1

% Class 0 (Red)
X2 = randn(num_samples/2, 2) - 2;  % Centered at (-2,-2)
Y2 = zeros(num_samples/2, 1);      % Class 0

% Combine dataset
X = [X1; X2];  % Features (Nx2)
Y = [Y1; Y2];  % Labels (Nx1)

% Shuffle data
idx = randperm(num_samples);
X = X(idx, :);
Y = Y(idx);

% Plot dataset
figure; hold on;
scatter(X(Y == 1, 1), X(Y == 1, 2), 'bo', 'filled'); % Class 1
scatter(X(Y == 0, 1), X(Y == 0, 2), 'ro', 'filled'); % Class 0
xlabel('Feature 1'); ylabel('Feature 2');
title('Synthetic Binary Classification Dataset');
legend('Class 1', 'Class 0');
grid on;
hold off;

% Define Logistic Regression parameters
alpha = 0.1;    % Learning rate
epochs = 1000;  % Number of iterations

% Train Logistic Regression model
% model = logistic_regression_train(X, Y, alpha, epochs);
model = logistic_regression_train_fast(X, Y, alpha, epochs);
disp('Training complete!');

% Predict function
Y_pred = round(1 ./ (1 + exp(-[ones(size(X,1),1), X] * model.w)));

% Compute accuracy
accuracy = mean(Y_pred == Y) * 100;
fprintf('Logistic Regression Accuracy: %.2f%%\n', accuracy);

% Create a mesh grid for visualization
[x1_grid, x2_grid] = meshgrid(linspace(min(X(:,1)), max(X(:,1)), 100), ...
                              linspace(min(X(:,2)), max(X(:,2)), 100));
X_grid = [x1_grid(:), x2_grid(:)];

% Compute predictions for the grid
Z = 1 ./ (1 + exp(-[ones(size(X_grid,1),1), X_grid] * model.w));
Z = reshape(Z, size(x1_grid));


% Plot decision boundary
figure; hold on;
scatter(X(Y == 1, 1), X(Y == 1, 2), 'bo'); % Class 1
scatter(X(Y == 0, 1), X(Y == 0, 2), 'bo'); % Class 0
xlabel('Feature 1'); ylabel('Feature 2');
title('Logistic Regression');
legend('Clusters', 'Location','SE');
grid on;
hold off;

% save_all_figs_OPTION('results/logistic_regression1','png',1)


% Plot decision boundary
figure; hold on;
scatter(X(Y == 1, 1), X(Y == 1, 2), 'bo'); % Class 1
scatter(X(Y == 0, 1), X(Y == 0, 2), 'ro'); % Class 0
contourf(x1_grid, x2_grid, Z, [-0.5 0.5], 'k', 'LineWidth', 0.8, 'FaceAlpha', 0.1); % Decision boundary
colormap([1 0.8 0.8; 0.8 0.8 1]);
xlabel('Feature 1'); ylabel('Feature 2');
title('Logistic Regression - Decision Boundary');
legend('Class 1', 'Class 0', 'Location','SE');
grid on;
hold off;

% save_all_figs_OPTION('results/logistic_regression2','png',1)


function model = logistic_regression_train_fast(X, Y, alpha, epochs)
    [N, M] = size(X);
    X = [ones(N, 1), X]; % Add bias term
    w = randn(M+1, 1) * 0.01; % Small random initialization
    
    for epoch = 1:epochs
        % Vectorized gradient descent
        y_pred = 1 ./ (1 + exp(-X * w));
        w = w - alpha * (X' * (y_pred - Y)) / N;
    end
    
    model.w = w;
end

function model = logistic_regression_train(X, Y, alpha, epochs)
    % Inputs:
    %   X - Feature matrix (NxM)
    %   Y - Target labels (Nx1) {0,1}
    %   alpha - Learning rate
    %   epochs - Number of training iterations
    % Output:
    %   model - Struct containing trained weights
    
    [N, M] = size(X);
    X = [ones(N, 1), X]; % Add bias term
    w = zeros(M+1, 1);   % Initialize weights
    
    for epoch = 1:epochs
        % Compute predictions using sigmoid function
        z = X * w;
        y_pred = 1 ./ (1 + exp(-z));
        
        % Compute gradient and update weights
        gradient = (X' * (y_pred - Y)) / N;
        w = w - alpha * gradient;
        
        % Compute loss (log loss)
        loss = -mean(Y .* log(y_pred) + (1 - Y) .* log(1 - y_pred));
        
        if mod(epoch, 100) == 0
            fprintf('Epoch %d, Loss: %.4f\n', epoch, loss);
        end
    end
    
    model.w = w;
end

