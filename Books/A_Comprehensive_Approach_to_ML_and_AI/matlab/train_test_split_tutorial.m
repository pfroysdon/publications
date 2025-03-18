close all; clear all; clc; 

% Generate synthetic dataset (100 samples, 3 features)
rng(42); % For reproducibility
X = randn(100, 3);  % Feature matrix (100x3)
Y = randi([0, 1], 100, 1);  % Binary class labels (100x1)

% Define train-test split ratio (e.g., 80% training, 20% testing)
train_ratio = 0.8;

% Apply train-test split
[X_train, Y_train, X_test, Y_test] = train_test_split(X, Y, train_ratio);

% Display dataset sizes
fprintf('Training set size: %d samples\n', size(X_train, 1));
fprintf('Testing set size: %d samples\n', size(X_test, 1));

% Plot training and testing data distribution
figure; hold on;
scatter3(X_train(:,1), X_train(:,2), X_train(:,3), 'bo', 'DisplayName', 'Train Data');
scatter3(X_test(:,1), X_test(:,2), X_test(:,3), 'ro', 'DisplayName', 'Test Data');
xlabel('Feature 1'); ylabel('Feature 2'); zlabel('Feature 3');
title('Train-Test Split Visualization');
legend;
grid on;
hold off;

% save_all_figs_OPTION('results/trainTestSplit','png',1)

function [X_train, Y_train, X_test, Y_test] = train_test_split(X, Y, train_ratio)
    % Splits dataset into training and testing sets.
    % Inputs:
    %   X - Feature matrix (NxM) [N samples, M features]
    %   Y - Target labels (Nx1) [N samples]
    %   train_ratio - Fraction of data to use for training (e.g., 0.8 for 80%)
    % Outputs:
    %   X_train - Training features
    %   Y_train - Training labels
    %   X_test - Testing features
    %   Y_test - Testing labels

    % Get number of samples
    N = size(X, 1);

    % Compute training size
    train_size = floor(train_ratio * N);

    % Shuffle data
    rand_indices = randperm(N);
    X = X(rand_indices, :);
    Y = Y(rand_indices, :);

    % Split data into training and testing sets
    X_train = X(1:train_size, :);
    Y_train = Y(1:train_size, :);
    X_test = X(train_size+1:end, :);
    Y_test = Y(train_size+1:end, :);
end
