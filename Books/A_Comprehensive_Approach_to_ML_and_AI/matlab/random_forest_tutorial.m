close all; clear all; clc;

% Generate synthetic dataset (two classes)
rng(42); % Ensure reproducibility
num_samples = 200;

% Class 1 (Blue)
X1 = randn(num_samples/2, 2) + 2;  % Cluster centered at (2,2)
Y1 = zeros(num_samples/2, 1);      % Class 0

% Class 2 (Red)
X2 = randn(num_samples/2, 2) - 2;  % Cluster centered at (-2,-2)
Y2 = ones(num_samples/2, 1);       % Class 1

% Combine the dataset
X = [X1; X2];  % Features (Nx2)
Y = [Y1; Y2];  % Labels (Nx1)

% Shuffle the data
idx = randperm(num_samples);
X = X(idx, :);
Y = Y(idx);

% Plot the dataset
figure; hold on;
scatter(X(Y == 0, 1), X(Y == 0, 2), 'bo'); % Class 0
scatter(X(Y == 1, 1), X(Y == 1, 2), 'bo'); % Class 1
xlabel('Feature 1'); ylabel('Feature 2');
title('Random Forest Classification');
legend('Clusters','Location','SE');
grid on;
hold off;

% save_all_figs_OPTION('results/random_forest1','png',1)

% Define Random Forest parameters
num_trees = 20;  % Number of trees
max_depth = 4;   % Maximum depth of each tree

% Train the Random Forest model
model = random_forest(X, Y, num_trees, max_depth);
disp('Training complete!');

% Predict on training data
Y_pred = predict_forest(model, X);

% Compute accuracy
accuracy = mean(Y_pred == Y) * 100;
fprintf('Model Accuracy: %.2f%%\n', accuracy);

% Create a mesh grid for visualization
[x1_grid, x2_grid] = meshgrid(linspace(min(X(:,1)), max(X(:,1)), 100), ...
                              linspace(min(X(:,2)), max(X(:,2)), 100));
X_grid = [x1_grid(:), x2_grid(:)];

% Predict labels for the grid
Y_grid = predict_forest(model, X_grid);
Y_grid = reshape(Y_grid, size(x1_grid));

% Plot decision boundary
figure; hold on;
scatter(X(Y == 0, 1), X(Y == 0, 2), 'ro'); % Class 0
scatter(X(Y == 1, 1), X(Y == 1, 2), 'bo'); % Class 1
% contourf(x1_grid, x2_grid, Y_grid, 'LineColor', 'none', 'FaceAlpha', 0.1);
contourf(x1_grid, x2_grid, Y_grid, 'LineWidth', 0.8, 'FaceAlpha', 0.1); % Decision boundary
colormap([1 0.8 0.8; 0.8 0.8 1]);
xlabel('Feature 1'); ylabel('Feature 2');
title('Random Forest Classification - Decision Boundary');
legend('Class 0','Class 1','Location','SE');
grid on;
hold off;

% save_all_figs_OPTION('results/random_forest2','png',1)


function model = random_forest(X, Y, num_trees, max_depth)
    % Optimized Random Forest using parallelization
    model.trees = cell(num_trees, 1);
    
    for t = 1:num_trees
        idx = randsample(size(X, 1), size(X, 1), true);
        X_sample = X(idx, :);
        Y_sample = Y(idx);
        
        model.trees{t} = fitrtree(X_sample, Y_sample, 'MaxDepth', max_depth);
    end
end

function Y_pred = predict_forest(model, X)
    num_trees = length(model.trees);
    predictions = zeros(size(X, 1), num_trees);

    for t = 1:num_trees
        predictions(:, t) = predict(model.trees{t}, X);  % Predict using each tree
    end

    % Aggregate predictions using majority voting (classification)
    Y_pred = mode(predictions, 2);
end

function tree = fitrtree(X, y, varargin)
% fitrtree Build a regression tree from scratch using the CART algorithm.
%
%   tree = fitrtree(X, y)
%
%   Inputs:
%       X - n x d matrix of features.
%       y - n x 1 vector of responses.
%
%   Optional Name-Value Parameters:
%       'MaxDepth'    - Maximum depth of the tree (default: 3)
%       'MinLeafSize' - Minimum number of samples required in a leaf (default: 5)
%
%   Output:
%       tree - A structure representing the regression tree with fields:
%           isLeaf      : true if the node is a leaf, false otherwise.
%           prediction  : mean of y at the node (if leaf).
%           feature     : index of feature used for splitting (if not leaf).
%           threshold   : threshold value used for splitting (if not leaf).
%           left        : left subtree (if not leaf).
%           right       : right subtree (if not leaf).
%           numSamples  : number of samples at the node.
%           mse         : mean squared error at the node.
%
%   Example:
%       tree = fitrtree(X, y, 'MaxDepth', 4, 'MinLeafSize', 10);

    % Parse optional parameters
    p = inputParser;
    addParameter(p, 'MaxDepth', 3);
    addParameter(p, 'MinLeafSize', 5);
    parse(p, varargin{:});
    maxDepth = p.Results.MaxDepth;
    minLeafSize = p.Results.MinLeafSize;
    
    % Build tree recursively starting at depth 1
    tree = buildTree(X, y, 1, maxDepth, minLeafSize);
end

function tree = buildTree(X, y, depth, maxDepth, minLeafSize)
    n = length(y);
    tree.numSamples = n;
    tree.prediction = mean(y);
    tree.mse = mean((y - tree.prediction).^2);
    
    % Stopping conditions: maximum depth reached, too few samples, or zero variance.
    if depth >= maxDepth || n <= minLeafSize || var(y) == 0
        tree.isLeaf = true;
        tree.feature = [];
        tree.threshold = [];
        tree.left = [];
        tree.right = [];
        return;
    end
    
    % Initialize variables for the best split
    bestMSE = Inf;
    bestFeature = [];
    bestThreshold = [];
    bestLeftIdx = [];
    bestRightIdx = [];
    
    [n, d] = size(X);
    
    % Try splitting on each feature
    for j = 1:d
        xj = X(:,j);
        uniqueVals = unique(xj);
        if length(uniqueVals) == 1
            continue; % cannot split on a constant feature
        end
        % Candidate thresholds: midpoints between consecutive unique values.
        thresholds = (uniqueVals(1:end-1) + uniqueVals(2:end)) / 2;
        for t = thresholds'
            leftIdx = xj <= t;
            rightIdx = xj > t;
            if sum(leftIdx) < minLeafSize || sum(rightIdx) < minLeafSize
                continue;
            end
            y_left = y(leftIdx);
            y_right = y(rightIdx);
            mse_left = mean((y_left - mean(y_left)).^2);
            mse_right = mean((y_right - mean(y_right)).^2);
            weightedMSE = (sum(leftIdx)*mse_left + sum(rightIdx)*mse_right) / n;
            if weightedMSE < bestMSE
                bestMSE = weightedMSE;
                bestFeature = j;
                bestThreshold = t;
                bestLeftIdx = leftIdx;
                bestRightIdx = rightIdx;
            end
        end
    end
    
    % If no valid split was found, make a leaf node.
    if isempty(bestFeature)
        tree.isLeaf = true;
        tree.feature = [];
        tree.threshold = [];
        tree.left = [];
        tree.right = [];
        return;
    end
    
    % Create internal node with best split.
    tree.isLeaf = false;
    tree.feature = bestFeature;
    tree.threshold = bestThreshold;
    % Recursively build left and right subtrees.
    tree.left = buildTree(X(bestLeftIdx, :), y(bestLeftIdx), depth + 1, maxDepth, minLeafSize);
    tree.right = buildTree(X(bestRightIdx, :), y(bestRightIdx), depth + 1, maxDepth, minLeafSize);
end


function y_pred = predict(tree, X)
% predict Predict responses for input data X using a regression tree.
%
%   y_pred = predict(tree, X)
%
%   Inputs:
%       tree - Regression tree structure (output of fitrtree).
%       X    - m x d matrix of features.
%
%   Output:
%       y_pred - m x 1 vector of predicted responses.
%
%   The function traverses the tree for each sample in X.

    m = size(X, 1);
    y_pred = zeros(m, 1);
    for i = 1:m
        y_pred(i) = traverseTree(tree, X(i, :));
    end
end

function pred = traverseTree(tree, x)
    % If the node is a leaf, return its prediction.
    if tree.isLeaf
        pred = tree.prediction;
    else
        % Otherwise, follow the branch based on the splitting feature and threshold.
        if x(tree.feature) <= tree.threshold
            pred = traverseTree(tree.left, x);
        else
            pred = traverseTree(tree.right, x);
        end
    end
end




