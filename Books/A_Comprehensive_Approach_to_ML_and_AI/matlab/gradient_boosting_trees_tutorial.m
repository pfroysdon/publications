% This tutorial demonstrates a simplified gradient boosting trees
% algorithm for binary classification using logistic loss.
%
% We:
%   1. Generate synthetic 2D data:
%         - Class 1: centered at (2,2)
%         - Class 0: centered at (-2,-2)
%   2. Train an ensemble of regression stumps sequentially by fitting
%      the negative gradient (y - p) of the logistic loss.
%   3. Compute predictions via an additive model and convert them to 
%      probabilities using the sigmoid function.
%   4. Visualize the decision boundary on a grid and report training accuracy.

close all; clear; clc; rng(1);  % For reproducibility

%% 1. Generate Synthetic Data
N = 100;
% Class 1: centered at (2,2)
X1 = randn(N,2) + 2;
% Class 0: centered at (-2,-2)
X0 = randn(N,2) - 2;
X = [X1; X0];
% Use labels 1 for class 1 and 0 for class 0
y = [ones(N,1); zeros(N,1)];

%% 2. Train Gradient Boosting Trees Model
T = 50;        % Number of boosting rounds
eta = 0.1;     % Learning rate
models = gradientBoostingTrain(X, y, T, eta);

%% 3. Compute Predictions on Training Data
y_pred = gradientBoostingPredict(X, models);
y_pred_class = double(y_pred >= 0.5);
train_acc = mean(y_pred_class == y);
fprintf('Gradient Boosting Trees Training Accuracy: %.2f%%\n', train_acc*100);

%% 4. Visualize the Decision Boundary
% Create a grid over the feature space.
x_min = min(X(:,1)) - 1; x_max = max(X(:,1)) + 1;
y_min = min(X(:,2)) - 1; y_max = max(X(:,2)) + 1;
[xx, yy] = meshgrid(linspace(x_min, x_max, 200), linspace(y_min, y_max, 200));
gridPoints = [xx(:), yy(:)];

preds_grid = gradientBoostingPredict(gridPoints, models);
preds_grid = reshape(preds_grid, size(xx));


% Plot the training data and decision boundary.
figure;
hold on;
scatter(X(y==1,1), X(y==1,2), 50, 'bo', 'DisplayName', 'Class 1');
scatter(X(y==0,1), X(y==0,2), 50, 'bo', 'DisplayName', 'Class 0');
title('Gradient Boosting Trees Classification');
xlabel('Feature 1'); ylabel('Feature 2');
legend('Clusters','Location','SE');
grid on;
hold off;

% save_all_figs_OPTION('results/gbt1','png',1)


% Plot the training data and decision boundary.
figure;
hold on;
scatter(X(y==1,1), X(y==1,2), 50, 'ro', 'DisplayName', 'Class 1');
scatter(X(y==0,1), X(y==0,2), 50, 'bo', 'DisplayName', 'Class 0');
% Contour at p = 0.5 shows the decision boundary.
% contourf(xx, yy, preds_grid, [0.5 0.5], 'LineWidth', 1, 'FaceAlpha', 0.1);
contourf(xx, yy, preds_grid, [-0.5 0.5], 'LineWidth', 0.8, 'FaceAlpha', 0.1); % Decision boundary
colormap([0.8 0.8 1; 1 0.8 0.8]);
title('Gradient Boosting Trees Classification - Decision Boundary');
xlabel('Feature 1'); ylabel('Feature 2');
legend('Class 0','Class 1','Location','SE');
grid on;
hold off;

% save_all_figs_OPTION('results/gbt2','png',1)


function models = gradientBoostingTrain(X, y, T, eta)
    % GRADIENTBOOSTINGTRAIN trains an ensemble of regression stumps using
    % gradient boosting with logistic loss.
    %
    % Inputs:
    %   X   - n x d data matrix.
    %   y   - n x 1 vector of binary labels (0 or 1).
    %   T   - number of boosting rounds.
    %   eta - learning rate.
    %
    % Output:
    %   models - a cell array of structures. Each structure has fields:
    %            .stump: the regression stump parameters.
    %            .coef:  the multiplier for the stump.
    
    n = size(X,1);
    F = zeros(n,1);  % Initialize additive model score F(x) = 0.
    models = cell(T,1);
    
    for t = 1:T
        % Compute current probabilities using the sigmoid function.
        p = sigmoid(F);
        % Negative gradient (working response) for logistic loss: r = y - p.
        r = y - p;
        
        % Fit a regression stump to the residuals r.
        stump = stumpRegTrain(X, r);
        h = stumpRegPredict(stump, X);
        
        % Optimal multiplier via least-squares line search: alpha = <r,h>/<h,h>.
        numerator   = sum(r .* h);
        denominator = sum(h .* h) + 1e-12;
        alpha = numerator / denominator;
        
        % Update the additive model.
        F = F + eta * alpha * h;
        
        % Store the stump and its (scaled) coefficient.
        models{t} = struct('stump', stump, 'coef', eta * alpha);
    end
end

function preds = gradientBoostingPredict(X, models)
    % GRADIENTBOOSTINGPREDICT computes predictions from the boosted model.
    %
    % Inputs:
    %   X      - n x d data matrix.
    %   models - cell array returned by gradientBoostingTrain.
    %
    % Output:
    %   preds  - n x 1 vector of predicted probabilities.
    
    n = size(X,1);
    F = zeros(n,1);  % Initialize additive score.
    T = length(models);
    
    for t = 1:T
        F = F + models{t}.coef * stumpRegPredict(models{t}.stump, X);
    end
    
    % Convert final scores to probabilities.
    preds = sigmoid(F);
end

function stump = stumpRegTrain(X, r)
    % STUMPREGTRAIN trains a regression stump to predict r.
    %
    % The stump splits on one feature at a threshold. Points with X(:,j) < threshold
    % are assigned a constant c1, and those with X(:,j) >= threshold are assigned c2.
    % The best split minimizes the sum of squared errors.
    %
    % Inputs:
    %   X - n x d data matrix.
    %   r - n x 1 vector of targets (residuals).
    %
    % Output:
    %   stump - structure with fields:
    %           .feature   - feature index used for splitting.
    %           .threshold - threshold value.
    %           .c1        - predicted value for left partition.
    %           .c2        - predicted value for right partition.
    
    [n, d] = size(X);
    bestLoss = Inf;
    bestFeature = 1;
    bestThreshold = 0;
    bestC1 = 0;
    bestC2 = 0;
    
    for j = 1:d
        thresholds = unique(X(:,j));
        for i = 1:length(thresholds)
            thresh = thresholds(i);
            left = X(:,j) < thresh;
            right = ~left;
            if sum(left) == 0 || sum(right) == 0
                continue;
            end
            c1 = mean(r(left));
            c2 = mean(r(right));
            loss = sum((r(left) - c1).^2) + sum((r(right) - c2).^2);
            if loss < bestLoss
                bestLoss = loss;
                bestFeature = j;
                bestThreshold = thresh;
                bestC1 = c1;
                bestC2 = c2;
            end
        end
    end
    
    stump.feature = bestFeature;
    stump.threshold = bestThreshold;
    stump.c1 = bestC1;
    stump.c2 = bestC2;
end

function yhat = stumpRegPredict(stump, X)
    % STUMPREGPREDICT returns predictions for data X using the regression stump.
    %
    % For each row in X, if X(:,feature) < threshold then the prediction is c1;
    % otherwise, it is c2.
    
    n = size(X,1);
    yhat = zeros(n,1);
    feature = stump.feature;
    thresh = stump.threshold;
    yhat(X(:,feature) < thresh) = stump.c1;
    yhat(X(:,feature) >= thresh) = stump.c2;
end

function s = sigmoid(z)
    % SIGMOID computes the logistic sigmoid function.
    s = 1 ./ (1 + exp(-z));
end
