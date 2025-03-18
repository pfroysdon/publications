% Demonstrates a simplified XGBoost-like approach using decision stumps
% on a 2D synthetic dataset. We:
%   1. Generate two Gaussian clusters for binary classification.
%   2. Train a gradient boosting ensemble under logistic loss.
%   3. Plot the decision boundary similar to the AdaBoost example.

close all; clc; rng(1); % For reproducibility

%% 1. Generate Synthetic Data
N = 100;
% Class 1: centered at (2,2)
X1 = randn(N,2) + 2;
% Class 0: centered at (-2,-2)
X0 = randn(N,2) - 2;
X = [X1; X0];
% For logistic classification, we use y in {0,1}
y = [ones(N,1); zeros(N,1)];

%% 2. Train XGBoost-like Model
T = 50;        % Number of boosting rounds
eta = 0.1;     % Learning rate
models = xgboostTrain(X, y, T, eta);

%% 3. Visualize Decision Boundary
% Create a grid of points covering the data region
x_min = min(X(:,1)) - 1;  x_max = max(X(:,1)) + 1;
y_min = min(X(:,2)) - 1;  y_max = max(X(:,2)) + 1;
[xx, yy] = meshgrid(linspace(x_min, x_max, 200), ...
                    linspace(y_min, y_max, 200));
gridPoints = [xx(:), yy(:)];

% Predict using the learned XGBoost model
preds = xgboostPredict(gridPoints, models);
preds = reshape(preds, size(xx));


% Plot data points
figure;
hold on;
scatter(X(y==1,1), X(y==1,2), 50, 'bo','DisplayName','Class 1');
scatter(X(y==0,1), X(y==0,2), 50, 'bo','DisplayName','Class 0');
title('XGBoost Classification');
xlabel('Feature 1'); ylabel('Feature 2');
legend('Clusters','Location','SE');
axis tight; grid on;
hold off;

% save_all_figs_OPTION('results/xgboost1','png',1)


% Plot data points
figure;
hold on;
scatter(X(y==1,1), X(y==1,2), 50, 'bo','DisplayName','Class 1');
scatter(X(y==0,1), X(y==0,2), 50, 'ro','DisplayName','Class 0');
% Plot the decision boundary via contour
% We define class 1 if preds >= 0.5, else class 0.
% contourf(xx, yy, preds, [0.5 0.5], 'k', 'LineWidth',1, 'LineColor','k');
contourf(xx, yy, preds, [-0.5 0.5], 'k', 'LineWidth', 0.8, 'FaceAlpha', 0.1); % Decision boundary
colormap([1 0.8 0.8; 0.8 0.8 1]);
title('XGBoost Classification - Decision Boundaries');
xlabel('Feature 1'); ylabel('Feature 2');
legend('Class 0','Class 1','Location','SE');
axis tight; grid on;
hold off;

% save_all_figs_OPTION('results/xgboost2','png',1)



function models = xgboostTrain(X, y, T, eta)
    % XGBOOSTTRAIN trains a simplified gradient boosting model for binary
    % classification with logistic loss. Base learners are regression stumps.
    %
    % Inputs:
    %   X   - (n x d) data matrix
    %   y   - (n x 1) labels in {0,1}
    %   T   - number of boosting rounds
    %   eta - learning rate
    %
    % Output:
    %   models - cell array of structures with fields:
    %       .stump   - the regression stump parameters
    %       .coef    - the multiplier for this stump (similar to alpha)
    %
    % We maintain an additive model: F(x) = sum_{t=1..T} [eta * h_t(x)],
    % where each h_t is a regression stump fit to the negative gradient of
    % the logistic loss at iteration t.

    n = size(X,1);
    % We'll store an additive score F for each data point (start at 0).
    F = zeros(n,1);
    
    models = cell(T,1);
    
    for t = 1:T
        % 1) Compute predictions p = sigmoid(F)
        p = sigmoid(F);
        
        % 2) Compute the negative gradient of logistic loss:
        %    logistic loss L = - [ y*log(p) + (1-y)*log(1-p) ]
        %    derivative w.r.t. F: grad = y - p   (but sign reversed => p - y).
        %    Actually for gradient boosting, we want residual = -(dL/dF) = y - p
        grad = y - p;  % n x 1 residual we want to fit with a regression stump
        
        % 3) Fit a regression stump to grad
        stump = stumpRegTrain(X, grad);
        
        % 4) Predict the fitted residuals
        h = stumpRegPredict(stump, X);
        
        % 5) We can also do a line search for the best multiplier; for simplicity,
        %    we skip or set multiplier=1. Let's do a quick line search:
        %    alpha = argmin sum_i [ (grad_i - alpha*h_i)^2 ] => alpha = <grad,h>/<h,h>
        numerator   = sum(grad .* h);
        denominator = sum(h .* h) + 1e-12;
        alpha = numerator / denominator;
        
        % 6) Update the additive model
        F = F + eta * alpha * h;
        
        % Store the stump and alpha
        models{t} = struct('stump', stump, 'coef', eta * alpha);
    end
end

function preds = xgboostPredict(X, models)
    % XGBOOSTPREDICT applies the learned gradient boosting model to data X.
    %   X       - (n x d) matrix
    %   models  - cell array from xgboostTrain
    %
    % Output:
    %   preds   - predicted labels in [0,1] (soft-prob thresholded at 0.5).
    %             Actually, we return the predicted probability here,
    %             so the caller can do preds >= 0.5 => class 1 else class 0.
    
    n = size(X,1);
    F = zeros(n,1);  % The additive score
    T = length(models);
    
    for t = 1:T
        stump = models{t}.stump;
        alpha = models{t}.coef;
        F = F + alpha * stumpRegPredict(stump, X);
    end
    
    % Convert final scores F to probabilities with sigmoid
    p = sigmoid(F);
    
    % Return probabilities (the plotting code uses contour at p=0.5)
    preds = p;
end

function stump = stumpRegTrain(X, r)
    % STUMPREGTRAIN: Train a regression stump that predicts a constant
    % value c1 for points X(:,feature) < threshold, and c2 otherwise.
    %
    % We find the split (feature, threshold) and constants (c1, c2) that
    % minimize sum of squared errors relative to the target r.
    
    [n, d] = size(X);
    
    bestFeature = 1;
    bestThreshold = 0;
    bestLoss = Inf;
    bestC1 = 0;
    bestC2 = 0;
    
    for j = 1:d
        % Sort by feature j
        [xjSorted, idx] = sort(X(:,j));
        rSorted = r(idx);
        
        % Possible thresholds are midpoints between consecutive unique x_j
        % We'll also consider a threshold below min or above max for completeness
        uniqueVals = unique(xjSorted);
        
        % We'll do a one-pass approach:
        %   Suffix sums to quickly evaluate c1,c2 for each threshold.
        % Let prefix sums of r, and prefix sums of r^2, etc.
        prefixSum = [0; cumsum(rSorted)];
        prefixSqSum = [0; cumsum(rSorted.^2)];
        
        for i = 1:length(uniqueVals)
            thresh = uniqueVals(i);
            % We find the index up to which x < thresh
            pos = find(xjSorted < thresh, 1, 'last');
            if isempty(pos)
                % All points are >= thresh
                pos = 0;
            end
            
            % For the left side: we want c1 = average of r in [1..pos]
            sumLeft  = prefixSum(pos+1);
            nLeft    = pos;
            if nLeft > 0
                c1 = sumLeft / nLeft;
            else
                c1 = 0;
            end
            
            % For the right side: average of r in [pos+1..n]
            sumRight = prefixSum(n+1) - sumLeft;
            nRight   = n - nLeft;
            if nRight > 0
                c2 = sumRight / nRight;
            else
                c2 = 0;
            end
            
            % Compute sum of squared errors
            % SSE = sum_{left}(r_i - c1)^2 + sum_{right}(r_i - c2)^2
            % Use prefix sums for speed
            sseLeft = prefixSqSum(pos+1) - 2*c1*sumLeft + nLeft*c1^2;
            sseRight = (prefixSqSum(n+1) - prefixSqSum(pos+1)) ...
                       - 2*c2*sumRight + nRight*c2^2;
            sse = sseLeft + sseRight;
            
            if sse < bestLoss
                bestLoss = sse;
                bestFeature = j;
                bestThreshold = thresh;
                bestC1 = c1;
                bestC2 = c2;
            end
        end
    end
    
    stump.feature   = bestFeature;
    stump.threshold = bestThreshold;
    stump.c1        = bestC1;
    stump.c2        = bestC2;
end

function yhat = stumpRegPredict(stump, X)
    % STUMPREGPREDICT: For a regression stump, predict c1 if X(:,feature) < threshold,
    % otherwise c2.
    j = stump.feature;
    thr = stump.threshold;
    c1 = stump.c1;
    c2 = stump.c2;
    
    yhat = zeros(size(X,1),1);
    mask = (X(:,j) < thr);
    yhat(mask) = c1;
    yhat(~mask) = c2;
end

function s = sigmoid(z)
    s = 1 ./ (1 + exp(-z));
end
