close all; clear all; clc; 

% Example usage:
rng(1); % For reproducibility
% Generate synthetic data for regression
X = randn(100,2);
y = sin(X(:,1)) + 0.5*X(:,2) + randn(100,1)*0.1;  % Some nonlinear function

M = 50;      % Number of boosting iterations
eta = 0.1;   % Learning rate
F = gradient_boosting_trees(X, y, M, eta);

% Compute predictions for training data
y_pred = F(X);

% Plot the true vs predicted values
figure;
scatter(y, y_pred, 'bo');
xlabel('True Values');
ylabel('Predicted Values');
title('Gradient Boosting Trees Regression');
grid on;

% save_all_figs_OPTION('results/gbt','png',1)


function F = gradient_boosting_trees(X, y, M, eta)
    % GRADIENTBOOSTINGTREES performs gradient boosting for regression.
    %   X: m x n data matrix (each row is a data point)
    %   y: m x 1 target vector
    %   M: number of boosting iterations
    %   eta: learning rate
    %   F: final predictive model (function handle)
    
    m = size(X,1);
    
    % Initialize model: for squared error loss, the best constant is the mean.
    F_pred = mean(y) * ones(m, 1);
    models = cell(M,1);
    gammas = zeros(M,1);
    
    for iter = 1:M
        % Compute residuals (for squared error loss, residuals are y - F_pred)
        residuals = y - F_pred;
        
        % Fit a simple regression tree to residuals (from scratch)
        tree = buildTree(X, residuals, 5, 3); % 5: maxDepth, 3: minLeaf (example parameters)
        models{iter} = tree;
        
        % Compute optimal gamma (for squared error, optimal gamma is 1)
        gamma = 1;  % For squared error loss, this is optimal.
        gammas(iter) = gamma;
        
        % Update prediction
        update = predictTree(tree, X);
        F_pred = F_pred + eta * gamma * update;
    end
    
    % Return final model as a function handle that aggregates predictions
    F = @(X_new) aggregatePrediction(X_new, models, gammas, eta, mean(y));
end

function tree = buildTree(X, y, maxDepth, minLeaf)
    % BUILDTREE builds a regression tree recursively.
    if (maxDepth == 0) || (length(y) < minLeaf) || (std(y)==0)
        tree.isLeaf = true;
        tree.prediction = mean(y);
        return;
    end
    
    [bestFeature, bestThreshold, bestGain] = chooseBestSplit(X, y);
    if bestGain <= 0
        tree.isLeaf = true;
        tree.prediction = mean(y);
        return;
    end
    
    tree.isLeaf = false;
    tree.feature = bestFeature;
    tree.threshold = bestThreshold;
    
    leftIdx = X(:, bestFeature) <= bestThreshold;
    rightIdx = ~leftIdx;
    
    tree.left = buildTree(X(leftIdx, :), y(leftIdx), maxDepth-1, minLeaf);
    tree.right = buildTree(X(rightIdx, :), y(rightIdx), maxDepth-1, minLeaf);
end

function [bestFeature, bestThreshold, bestGain] = chooseBestSplit(X, y)
    % CHOOSEBESTSPLIT finds the best feature and threshold using squared error reduction.
    [numSamples, numFeatures] = size(X);
    bestGain = -inf;
    bestFeature = 0;
    bestThreshold = 0;
    currentError = sum((y - mean(y)).^2);
    
    for feature = 1:numFeatures
        thresholds = unique(X(:, feature));
        for t = thresholds'
            leftIdx = X(:, feature) <= t;
            rightIdx = ~leftIdx;
            if isempty(y(leftIdx)) || isempty(y(rightIdx))
                continue;
            end
            errorLeft = sum((y(leftIdx) - mean(y(leftIdx))).^2);
            errorRight = sum((y(rightIdx) - mean(y(rightIdx))).^2);
            gain = currentError - (length(y(leftIdx))/numSamples * errorLeft + ...
                   length(y(rightIdx))/numSamples * errorRight);
            if gain > bestGain
                bestGain = gain;
                bestFeature = feature;
                bestThreshold = t;
            end
        end
    end
end

function y_pred = predictTree(tree, X)
    % PREDICTTREE predicts responses for X using the regression tree.
    m = size(X,1);
    y_pred = zeros(m,1);
    for i = 1:m
        y_pred(i) = traverseTree(tree, X(i,:));
    end
end

function prediction = traverseTree(tree, x)
    if tree.isLeaf
        prediction = tree.prediction;
    else
        if x(tree.feature) <= tree.threshold
            prediction = traverseTree(tree.left, x);
        else
            prediction = traverseTree(tree.right, x);
        end
    end
end

function pred = aggregatePrediction(X_new, models, gammas, eta, initPrediction)
    % AGGREGATEPREDICTION aggregates the predictions of all trees.
    m = size(X_new,1);
    pred = initPrediction * ones(m,1);
    for i = 1:length(models)
        update = predictTree(models{i}, X_new);
        pred = pred + eta * gammas(i) * update;
    end
end