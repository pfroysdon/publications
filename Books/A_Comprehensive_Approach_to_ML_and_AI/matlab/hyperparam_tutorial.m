close all; clear all; clc;

% Generate synthetic data for regression
n = 200;
X = linspace(0,10,n)';
y = sin(X) + 0.5*randn(n,1);

% Split data into training and test (holdout set)
testRatio = 0.2;
[X_train, y_train, X_test, y_test] = holdoutSplit(X, y, testRatio);

% Define hyperparameter grid for a simple ridge regression model (lambda)
lambdaGrid = logspace(-4, 2, 10);

% Perform 5-fold cross-validation to select the best lambda
k = 5;
cvErrors = zeros(length(lambdaGrid),1);
for i = 1:length(lambdaGrid)
    lambda = lambdaGrid(i);
    cvErrors(i) = kFoldCV(X_train, y_train, k, lambda);
end
[~, bestIdx] = min(cvErrors);
bestLambda = lambdaGrid(bestIdx);
fprintf('Best lambda from grid search: %.4f\n', bestLambda);

% Train final model on full training set using bestLambda
theta = ridgeRegression(X_train, y_train, bestLambda);
y_pred = X_test * theta;
testError = mean((y_test - y_pred).^2);
fprintf('Test MSE: %.4f\n', testError);

% Plot validation curve
figure;
semilogx(lambdaGrid, cvErrors, 'bo-', 'LineWidth', 2);
xlabel('Lambda');
ylabel('Cross-Validation MSE');
title('Validation Curve');
grid on;

% Compute and plot learning curves
[trainErrors, valErrors] = learningCurves(X_train, y_train, k, bestLambda);
figure;
plot(1:length(trainErrors), trainErrors, 'b-', 'LineWidth', 2); hold on;
plot(1:length(valErrors), valErrors, 'r-', 'LineWidth', 2);
xlabel('Training set size');
ylabel('MSE');
title('Learning Curves');
legend('Training Error', 'Validation Error');
grid on;

% save_all_figs_OPTION('results/hyperparam','png',1)

%% Function: Holdout Split
function [X_train, y_train, X_test, y_test] = holdoutSplit(X, y, testRatio)
    n = length(y);
    idx = randperm(n);
    nTest = round(testRatio * n);
    testIdx = idx(1:nTest);
    trainIdx = idx(nTest+1:end);
    X_train = X(trainIdx,:);
    y_train = y(trainIdx,:);
    X_test = X(testIdx,:);
    y_test = y(testIdx,:);
end

%% Function: k-Fold Cross-Validation for Ridge Regression
function cvError = kFoldCV(X, y, k, lambda)
    n = length(y);
    indices = crossvalind('Kfold', n, k);
    errors = zeros(k,1);
    for i = 1:k
        testIdx = (indices == i);
        trainIdx = ~testIdx;
        X_train = X(trainIdx,:);
        y_train = y(trainIdx,:);
        X_val = X(testIdx,:);
        y_val = y(testIdx,:);
        theta = ridgeRegression(X_train, y_train, lambda);
        y_pred = X_val * theta;
        errors(i) = mean((y_val - y_pred).^2);
    end
    cvError = mean(errors);
end

%% Function: Ridge Regression (Closed-Form)
function theta = ridgeRegression(X, y, lambda)
    % Closed-form solution for ridge regression: theta = (X'X + lambda*I)^(-1) X'y
    [n, p] = size(X);
    I = eye(p);
    theta = (X'*X + lambda * I) \ (X'*y);
end

%% Function: Learning Curves
function [trainErrors, valErrors] = learningCurves(X, y, k, lambda)
    n = length(y);
    % Define a set of training sizes
    sizes = round(linspace(floor(0.1*n), n, 10));
    trainErrors = zeros(length(sizes),1);
    valErrors = zeros(length(sizes),1);
    indices = crossvalind('Kfold', n, k);
    for s = 1:length(sizes)
        sizeTrain = sizes(s);
        % Randomly sample sizeTrain examples for training and use remaining for validation
        idx = randperm(n);
        trainIdx = idx(1:sizeTrain);
        valIdx = idx(sizeTrain+1:end);
        X_train = X(trainIdx,:);
        y_train = y(trainIdx,:);
        X_val = X(valIdx,:);
        y_val = y(valIdx,:);
        theta = ridgeRegression(X_train, y_train, lambda);
        y_pred_train = X_train * theta;
        y_pred_val = X_val * theta;
        trainErrors(s) = mean((y_train - y_pred_train).^2);
        valErrors(s) = mean((y_val - y_pred_val).^2);
    end
end

function idx = crossvalind(method, n, k)
% crossvalind Generate cross validation indices for k-fold cross validation.
%
%   idx = crossvalind('Kfold', n, k)
%
%   Inputs:
%       method - Cross-validation method. Only 'Kfold' is supported.
%       n      - Total number of observations (scalar).
%       k      - Number of folds (scalar).
%
%   Output:
%       idx    - n-by-1 vector of fold assignments, where each element is an
%                integer between 1 and k.
%
%   Example:
%       idx = crossvalind('Kfold', 100, 5);
%
%   This function randomly permutes the indices from 1 to n and assigns them
%   to k folds as equally as possible.

    % Check method
    if ~strcmpi(method, 'Kfold')
        error('Unsupported cross-validation method: %s', method);
    end

    % Check that k is less than or equal to n
    if k > n
        error('Number of folds k (%d) cannot exceed number of observations n (%d).', k, n);
    end

    % Initialize the index vector
    idx = zeros(n, 1);

    % Randomly permute the indices 1:n
    perm = randperm(n);

    % Compute fold sizes. If n is not divisible by k, the first 'remainder' folds
    % get one extra sample.
    fold_sizes = repmat(floor(n/k), k, 1);
    remainder = mod(n, k);
    fold_sizes(1:remainder) = fold_sizes(1:remainder) + 1;

    % Assign fold numbers to the permuted indices.
    start_idx = 1;
    for fold = 1:k
        end_idx = start_idx + fold_sizes(fold) - 1;
        idx(perm(start_idx:end_idx)) = fold;
        start_idx = end_idx + 1;
    end
end
