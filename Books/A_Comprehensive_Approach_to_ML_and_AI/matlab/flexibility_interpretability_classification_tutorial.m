function flexibility_vs_interpretability_classification
    % This script demonstrates the trade-off between flexibility and
    % interpretability using a classification example.
    %
    % We generate synthetic 2D data where the true decision boundary is circular.
    % Then, we apply:
    %   1. Logistic Regression (a linear, interpretable model).
    %   2. k-Nearest Neighbors (k-NN) with k = 5 (a flexible, non-parametric model).
    %
    % Logistic regression produces a straight-line decision boundary,
    % while k-NN is capable of capturing the circular structure.
    
    %% 1. Generate Synthetic Data
    rng(1); % For reproducibility
    N = 200;
    % Generate points uniformly in the square [-1,1] x [-1,1]
    X = 2*rand(N,2) - 1;
    % Label = 1 if the point lies inside a circle of radius 0.5 (centered at 0,0), 0 otherwise
    y = double(sum(X.^2, 2) < 0.5^2);
    
    %% 2. Fit Logistic Regression (Interpretable Model)
    % Add an intercept term
    X_lr = addIntercept(X);
    lr_rate = 0.1;
    num_iters = 1000;
    [w, losses] = logisticRegression(X_lr, y, lr_rate, num_iters);
    
    % Display the logistic regression coefficients
    fprintf('Logistic Regression Coefficients (Interpretable):\n');
    disp(w);
    
    %% 3. Fit k-NN Classifier (Flexible Model)
    k = 5;
    % (k-NN is a lazy learner that uses training data directly.)
    
    %% 4. Plot Decision Boundaries
    % Create a grid of points for visualization
    [xx, yy] = meshgrid(linspace(-1,1,100), linspace(-1,1,100));
    grid_points = [xx(:), yy(:)];
    
    % Logistic Regression predictions (probabilities)
    grid_points_lr = addIntercept(grid_points);
    preds_lr = 1 ./ (1 + exp(-grid_points_lr * w));
    preds_lr = reshape(preds_lr, size(xx));
    
    % k-NN predictions
    preds_knn = knnClassifier(X, y, grid_points, k);
    preds_knn = reshape(preds_knn, size(xx));
    
    %% 5. Visualization
    figure('Position',[100 100 1200 500]);
    
    % Subplot for Logistic Regression
    subplot(1,2,1);
    hold on;
    % Draw the decision boundary (probability 0.5 contour)
    contour(xx, yy, preds_lr, [0.5 0.5], 'r', 'LineWidth',2);
    % Plot data: blue for class 0, green for class 1
    scatter(X(y==0,1), X(y==0,2), 50, 'b', 'filled');
    scatter(X(y==1,1), X(y==1,2), 50, 'g', 'filled');
    title('Logistic Regression (Interpretable)');
    xlabel('x_1'); ylabel('x_2');
    legend('Decision Boundary','Class 0','Class 1','Location','Best');
    grid on;
    hold off;
    
    % Subplot for k-NN
    subplot(1,2,2);
    hold on;
    contour(xx, yy, preds_knn, [0.5 0.5], 'r', 'LineWidth',2);
    scatter(X(y==0,1), X(y==0,2), 50, 'b', 'filled');
    scatter(X(y==1,1), X(y==1,2), 50, 'g', 'filled');
    title('k-NN (Flexible, k = 5)');
    xlabel('x_1'); ylabel('x_2');
    legend('Decision Boundary','Class 0','Class 1','Location','Best');
    grid on;
    hold off;
    
    fprintf('\nNote:\n');
    fprintf(' - Logistic Regression yields a linear (straight-line) decision boundary that is simple and interpretable.\n');
    fprintf(' - k-NN is more flexible and captures the circular boundary, but its decision rule (based on training examples) is less transparent.\n');
end

%% Helper function: Logistic Regression via Gradient Descent
function [w, losses] = logisticRegression(X, y, lr, num_iters)
    [n, d] = size(X);
    w = zeros(d,1);         % Initialize weights
    losses = zeros(num_iters,1);
    for iter = 1:num_iters
        logits = X * w;
        preds = 1 ./ (1 + exp(-logits));  % Sigmoid function
        % Binary cross-entropy loss
        loss = -mean(y .* log(preds+1e-8) + (1-y) .* log(1-preds+1e-8));
        losses(iter) = loss;
        grad = (X' * (preds - y)) / n;
        w = w - lr * grad;
    end
end

%% Helper function: k-NN Classifier (from scratch)
function preds = knnClassifier(X_train, y_train, X_test, k)
    n_test = size(X_test,1);
    preds = zeros(n_test,1);
    for i = 1:n_test
        % Compute Euclidean distances between the test point and all training points
        dists = sum((X_train - X_test(i,:)).^2, 2);
        [~, idx] = sort(dists);
        nearest_labels = y_train(idx(1:k));
        preds(i) = mode(nearest_labels); % Majority vote
    end
end

%% Helper function: Add Intercept Term
function X_intercept = addIntercept(X)
    X_intercept = [ones(size(X,1),1), X];
end
