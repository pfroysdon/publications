% This tutorial demonstrates a decision tree classifier built entirely from
% scratch. We generate a 2D dataset with two classes, recursively build a tree
% using the Gini impurity criterion, and visualize the decision boundary.

close all; clear; clc;
rng(1);  % For reproducibility

%% 1. Generate Synthetic Data
N = 100;
% Class +1: centered at (2,2)
X1 = randn(N,2) + 2;
% Class -1: centered at (-2,-2)
X2 = randn(N,2) - 2;
X = [X1; X2];
y = [ones(N,1); -ones(N,1)];  % Labels: +1 and -1

%% 2. Build Decision Tree from Scratch
maxDepth = 3;  % Limit tree depth for simplicity (similar to MaxNumSplits = 5)
tree = buildTree(X, y, 0, maxDepth);

%% 3. Visualize the Decision Boundary
% Create a grid over the feature space.
x_min = min(X(:,1)) - 1; x_max = max(X(:,1)) + 1;
y_min = min(X(:,2)) - 1; y_max = max(X(:,2)) + 1;
[xGrid, yGrid] = meshgrid(linspace(x_min, x_max, 200), linspace(y_min, y_max, 200));
gridPoints = [xGrid(:), yGrid(:)];

% Predict using the tree
preds = predictTree(tree, gridPoints);
preds = reshape(preds, size(xGrid));

% Plot data and decision regions.
figure;
hold on;
scatter(X(y==1,1), X(y==1,2), 50, 'bo', 'DisplayName', 'Class +1');
scatter(X(y==-1,1), X(y==-1,2), 50, 'bo', 'DisplayName', 'Class -1');
title('Decision Tree Classification');
xlabel('Feature 1'); ylabel('Feature 2');
legend('Clusters','Location','SE');
grid on;
hold off;

% save_all_figs_OPTION('results/decision_tree1','png',1)

% Plot data and decision regions.
figure;
hold on;
scatter(X(y==1,1), X(y==1,2), 50, 'bo', 'DisplayName', 'Class +1');
scatter(X(y==-1,1), X(y==-1,2), 50, 'ro', 'DisplayName', 'Class -1');
contourf(xGrid, yGrid, preds, [-1 0 1], 'LineWidth', 0.8, 'FaceAlpha', 0.1); % Decision boundary
colormap([1 0.8 0.8; 0.8 0.8 1]);  % Red-ish for class -1, blue-ish for class +1
title('Decision Tree Classification - Decision Boundary');
xlabel('Feature 1'); ylabel('Feature 2');
legend('Class 0','Class 1','Location','SE');
grid on;
hold off;

% save_all_figs_OPTION('results/decision_tree2','png',1)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                   Subfunctions for Decision Tree
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function tree = buildTree(X, y, depth, maxDepth)
    % buildTree recursively builds a decision tree classifier.
    %
    % Inputs:
    %   X       - n x d data matrix.
    %   y       - n x 1 vector of labels (+1 or -1).
    %   depth   - current depth of the tree.
    %   maxDepth- maximum allowed depth.
    %
    % Output:
    %   tree    - a struct representing the decision tree.
    %
    % The tree struct has the fields:
    %   .isLeaf      - boolean flag indicating if the node is a leaf.
    %   .prediction  - predicted class if node is a leaf.
    %   .feature     - feature index for splitting.
    %   .threshold   - threshold value for the split.
    %   .left, .right- subtrees (if not a leaf).
    
    % Stopping criteria: all labels identical, max depth reached, or too few samples.
    if (all(y == y(1))) || (depth >= maxDepth) || (size(X,1) < 2)
        tree.isLeaf = true;
        tree.prediction = majorityVote(y);
        return;
    end
    
    % Find best split using Gini impurity.
    [bestFeature, bestThreshold, bestImpurity, bestSplit] = findBestSplit(X, y);
    
    % If no valid split is found, create a leaf.
    if isempty(bestFeature)
        tree.isLeaf = true;
        tree.prediction = majorityVote(y);
        return;
    end
    
    % Create internal node.
    tree.isLeaf = false;
    tree.feature = bestFeature;
    tree.threshold = bestThreshold;
    
    % Partition data based on the best split.
    leftIndices = bestSplit;
    rightIndices = ~bestSplit;
    
    % If one partition is empty, make this node a leaf.
    if isempty(find(leftIndices,1)) || isempty(find(rightIndices,1))
        tree.isLeaf = true;
        tree.prediction = majorityVote(y);
        return;
    end
    
    tree.left = buildTree(X(leftIndices, :), y(leftIndices), depth+1, maxDepth);
    tree.right = buildTree(X(rightIndices, :), y(rightIndices), depth+1, maxDepth);
end

function [bestFeature, bestThreshold, bestImpurity, bestSplit] = findBestSplit(X, y)
    % findBestSplit searches over features and thresholds to find the split
    % that minimizes the weighted Gini impurity.
    %
    % Outputs:
    %   bestFeature   - index of the best feature.
    %   bestThreshold - threshold value for the best split.
    %   bestImpurity  - weighted impurity of the best split.
    %   bestSplit     - logical vector indicating the left split (X(:,feature) < threshold).
    
    [n, d] = size(X);
    bestImpurity = Inf;
    bestFeature = [];
    bestThreshold = [];
    bestSplit = [];
    
    for j = 1:d
        values = unique(X(:,j));
        for i = 1:length(values)
            threshold = values(i);
            split = X(:,j) < threshold;
            if sum(split) == 0 || sum(~split) == 0
                continue;  % Skip invalid splits.
            end
            impurityLeft = giniImpurity(y(split));
            impurityRight = giniImpurity(y(~split));
            weightedImpurity = (sum(split)/n)*impurityLeft + (sum(~split)/n)*impurityRight;
            if weightedImpurity < bestImpurity
                bestImpurity = weightedImpurity;
                bestFeature = j;
                bestThreshold = threshold;
                bestSplit = split;
            end
        end
    end
end

function gini = giniImpurity(y)
    % giniImpurity computes the Gini impurity for binary labels.
    n = length(y);
    if n == 0
        gini = 0;
    else
        p = sum(y == 1) / n;
        gini = 1 - (p^2 + (1 - p)^2);
    end
end

function vote = majorityVote(y)
    % majorityVote returns the majority class from the labels.
    if sum(y == 1) >= sum(y == -1)
        vote = 1;
    else
        vote = -1;
    end
end

function preds = predictTree(tree, X)
    % predictTree applies the decision tree to each row of X.
    n = size(X,1);
    preds = zeros(n,1);
    for i = 1:n
        preds(i) = traverseTree(tree, X(i,:));
    end
end

function label = traverseTree(tree, x)
    % traverseTree recursively traverses the tree for a single sample.
    if tree.isLeaf
        label = tree.prediction;
    else
        if x(tree.feature) < tree.threshold
            label = traverseTree(tree.left, x);
        else
            label = traverseTree(tree.right, x);
        end
    end
end
