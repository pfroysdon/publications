close all; clear all; clc; 

% AdaBoostExample demonstrates AdaBoost with decision stumps on a 2D dataset.
%
% It generates two classes of data, trains AdaBoost, and plots the
% decision boundary.

% -------------------------------
% Generate Synthetic Data
% -------------------------------
rng(1);  % For reproducibility
N = 100;
% Class +1: centered at (2,2)
X1 = randn(N,2) + 2;
% Class -1: centered at (-2,-2)
X2 = randn(N,2) - 2;
X = [X1; X2];
y = [ones(N,1); -ones(N,1)];

% -------------------------------
% Train AdaBoost
% -------------------------------
T = 50;  % Number of weak classifiers
[stumps, alphas] = adaboostTrain(X, y, T);

% -------------------------------
% Visualize Decision Boundary
% -------------------------------
% Create a grid of points covering the data region
x_min = min(X(:,1)) - 1;
x_max = max(X(:,1)) + 1;
y_min = min(X(:,2)) - 1;
y_max = max(X(:,2)) + 1;
[xGrid, yGrid] = meshgrid(linspace(x_min, x_max, 200), linspace(y_min, y_max, 200));
gridPoints = [xGrid(:), yGrid(:)];

% Predict using the AdaBoost classifier
preds = adaboostPredict(gridPoints, stumps, alphas);
preds = reshape(preds, size(xGrid));

figure;
scatter(X(y==1,1), X(y==1,2), 50, 'bo');
hold on;
scatter(X(y==-1,1), X(y==-1,2), 50, 'bo');
title('AdaBoost Classification');
xlabel('Feature 1'); ylabel('Feature 2');
hold off;
legend('Clusters','Location','SE');

% save_all_figs_OPTION('results/adaboost1','png',1)


figure;
scatter(X(y==1,1), X(y==1,2), 50, 'bo');
hold on;
scatter(X(y==-1,1), X(y==-1,2), 50, 'ro');
% contourf(xGrid, yGrid, preds, 'k', 'LineWidth', 1, 'FaceAlpha', 0.1); % Decision boundary
contourf(xGrid, yGrid, preds, 'LineWidth', 0.8, 'FaceAlpha', 0.1); % Decision boundary
colormap([1 0.8 0.8; 0.8 0.8 1]);
title('AdaBoost Classification - Decision Boundaries');
xlabel('Feature 1'); ylabel('Feature 2');
hold off;
legend('Class 0','Class 1','Location','SE');

% save_all_figs_OPTION('results/adaboost2','png',1)


function [stumps, alphas] = adaboostTrain(X, y, T)
%  adaboostTrain: Trains AdaBoost with decision stumps.
%  Input:
%       X - n x d data matrix.
%       y - n x 1 labels in {-1,1}.
%       T - number of boosting iterations.
%  Output:
%       stumps - a cell array of decision stump structures.
%       alphas - a T x 1 vector of weak classifier weights.
    n = size(X,1);
    D = ones(n,1) / n;  % Initialize sample weights uniformly.
    stumps = cell(T, 1);
    alphas = zeros(T, 1);
    
    for t = 1:T
        % Train a decision stump with the current weights
        [stump, error, pred] = decisionStump(X, y, D);
        
        % Avoid division by zero (or log of zero)
        error = max(error, 1e-10);
        alpha = 0.5 * log((1 - error) / error);
        
        stumps{t} = stump;
        alphas(t) = alpha;
        
        % Update the weights: Increase weights on misclassified points.
        D = D .* exp(-alpha * y .* pred);
        D = D / sum(D);  % Normalize to sum to 1.
    end
end

function [bestStump, bestError, bestPred] = decisionStump(X, y, D)
%  decisionStump: Finds the best decision stump for the given data.
%  The decision stump is a one-level decision tree of the form:
%     if (feature < threshold) then predict p else predict -p,
%  where p = 1 or -1.
%
%  Input:
%       X - n x d data matrix.
%       y - n x 1 labels in {-1,1}.
%       D - n x 1 vector of sample weights.
%  Output:
%       bestStump - a structure with fields:
%                   .feature   (the feature index)
%                   .threshold (the threshold value)
%                   .polarity  (1 or -1)
%       bestError - the weighted error of this stump.
%       bestPred  - the predictions (n x 1) on X made by this stump.
    [n, d] = size(X);
    bestError = Inf;
    bestStump = struct('feature', 0, 'threshold', 0, 'polarity', 1);
    bestPred = zeros(n,1);
    
    % Loop over each feature
    for j = 1:d
        % Get all unique values in feature j to consider as thresholds
        thresholds = unique(X(:,j));
        for i = 1:length(thresholds)
            thresh = thresholds(i);
            % Try both possible polarity assignments: 1 and -1.
            for polarity = [1, -1]
                % Make predictions using the decision rule:
                % if polarity == 1: predict +1 when X(:,j) < thresh, else -1.
                % if polarity == -1: predict -1 when X(:,j) < thresh, else +1.
                if polarity == 1
                    pred = ones(n, 1);
                    pred(X(:,j) >= thresh) = -1;
                else
                    pred = -ones(n, 1);
                    pred(X(:,j) >= thresh) = 1;
                end
                
                % Compute the weighted error
                err = sum(D .* (pred ~= y));
                if err < bestError
                    bestError = err;
                    bestStump.feature = j;
                    bestStump.threshold = thresh;
                    bestStump.polarity = polarity;
                    bestPred = pred;
                end
            end
        end
    end
end

function predictions = adaboostPredict(X, stumps, alphas)
%  adaboostPredict: Uses the AdaBoost ensemble to make predictions.
%
%  Input:
%       X       - n x d data matrix of points to classify.
%       stumps  - cell array of decision stump structures.
%       alphas  - vector of weights for the stumps.
%  Output:
%       predictions - n x 1 vector of predicted labels in {-1,1}.
n = size(X, 1);
    T = length(stumps);
    agg = zeros(n,1);  % This will hold the weighted vote sum.
    
    for t = 1:T
        stump = stumps{t};
        feature = stump.feature;
        thresh = stump.threshold;
        polarity = stump.polarity;
        
        if polarity == 1
            pred = ones(n, 1);
            pred(X(:,feature) >= thresh) = -1;
        else
            pred = -ones(n, 1);
            pred(X(:,feature) >= thresh) = 1;
        end
        agg = agg + alphas(t) * pred;
    end
    
    predictions = sign(agg);
    % If any prediction is exactly 0, assign it to +1.
    predictions(predictions == 0) = 1;
end
