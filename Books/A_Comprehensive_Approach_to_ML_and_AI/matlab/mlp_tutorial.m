close all; clear all; clc;

% Generate synthetic dataset (2 classes)
rng(42); % For reproducibility
num_samples = 200;

% Class 1 (Blue)
X1 = randn(num_samples/2, 2) + 1;
Y1 = zeros(num_samples/2, 1);

% Class 2 (Red)
X2 = randn(num_samples/2, 2) - 1;
Y2 = ones(num_samples/2, 1);

% Combine the dataset
X = [X1; X2];  % Features (Nx2)
Y = [Y1; Y2];  % Labels (Nx1)

% Shuffle data
idx = randperm(num_samples);
X = X(idx, :);
Y = Y(idx, :);

% Plot the dataset
figure; hold on;
scatter(X(Y == 0, 1), X(Y == 0, 2), 'bo'); % Class 0
scatter(X(Y == 1, 1), X(Y == 1, 2), 'ro'); % Class 1
xlabel('Feature 1'); ylabel('Feature 2');
title('Synthetic Binary Classification Dataset');
legend('Class 0', 'Class 1');
grid on;
hold off;

% Normalize input features
X = (X - mean(X)) ./ std(X); 

% Define MLP parameters
num_hidden = 5;  % Number of hidden neurons
alpha = 0.1;     % Learning rate
epochs = 1000;   % Number of training iterations

% Train MLP
model = mlp_train(X, Y, num_hidden, alpha, epochs);
disp('Training complete!');

% Predict on training data
Y_pred = mlp_predict(model, X);

% Compute accuracy
accuracy = mean(Y_pred == Y) * 100;
fprintf('Model Accuracy: %.2f%%\n', accuracy);

% Create a mesh grid for visualization
[x1_grid, x2_grid] = meshgrid(linspace(min(X(:,1)), max(X(:,1)), 100), ...
                              linspace(min(X(:,2)), max(X(:,2)), 100));
X_grid = [x1_grid(:), x2_grid(:)];

% Predict labels for the grid
Y_grid = mlp_predict(model, X_grid);
Y_grid = reshape(Y_grid, size(x1_grid));

% Plot decision boundary
figure; hold on;
scatter(X(Y == 0, 1), X(Y == 0, 2), 'ro'); % Class 0
scatter(X(Y == 1, 1), X(Y == 1, 2), 'bo'); % Class 1
% contourf(x1_grid, x2_grid, Y_grid, 'LineColor', 'none', 'FaceAlpha', 0.1);
contourf(x1_grid, x2_grid, Y_grid, 'LineWidth', 0.8, 'FaceAlpha', 0.1); % Decision boundary
colormap([1 0.8 0.8; 0.8 0.8 1]);
xlabel('Feature 1'); ylabel('Feature 2');
title('MLP Decision Boundary');
legend('Class 0', 'Class 1', 'Location','SE');
grid on;
hold off;

% save_all_figs_OPTION('results/mlp','png',1)



function model = mlp_train(X, Y, num_hidden, alpha, epochs)
    % Optimized MLP training using vectorized operations
    [N, M] = size(X);
    
    W1 = randn(M, num_hidden) * sqrt(2/M);
    b1 = zeros(1, num_hidden);
    W2 = randn(num_hidden, 1) * sqrt(2/num_hidden);
    b2 = 0;
    
    for epoch = 1:epochs
        % Forward propagation
        A1 = max(0, X * W1 + b1);
        A2 = 1 ./ (1 + exp(-A1 * W2 - b2));
        
        % Backpropagation
        dZ2 = A2 - Y;
        dW2 = (A1' * dZ2) / N;
        db2 = mean(dZ2);
        dA1 = dZ2 * W2';
        dZ1 = dA1 .* (A1 > 0);
        dW1 = (X' * dZ1) / N;
        db1 = mean(dZ1);
        
        % Weight updates
        W1 = W1 - alpha * dW1;
        b1 = b1 - alpha * db1;
        W2 = W2 - alpha * dW2;
        b2 = b2 - alpha * db2;
    end
    
    model.W1 = W1;
    model.W2 = W2;
end

function Y_pred = mlp_predict(model, X)
    % Forward pass
    A1 = max(0, X * model.W1); % ReLU activation
    A2 = 1 ./ (1 + exp(-A1 * model.W2)); % Sigmoid activation
    
    % Convert probabilities to binary labels (Threshold = 0.5)
    Y_pred = A2 >= 0.5;
end

