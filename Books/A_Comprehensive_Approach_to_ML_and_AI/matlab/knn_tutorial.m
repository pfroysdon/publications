close all; clear all; clc; 

% Generate synthetic dataset
rng(42);
X = [randn(50,2) + 2; randn(50,2) - 2]; % Two Gaussian clusters
Y = [ones(50,1); zeros(50,1)]; % Class labels (1 or 0)

% Split data into training and testing sets
train_ratio = 0.8;
train_size = floor(train_ratio * size(X,1));
X_train = X(1:train_size, :);
Y_train = Y(1:train_size);
X_test = X(train_size+1:end, :);
Y_test = Y(train_size+1:end);

% Train and predict with KNN
K = 5;
Y_pred = knn_classify_fast(X_train, Y_train, X_test, K);

% Compute accuracy
accuracy = mean(Y_pred == Y_test) * 100;
fprintf('KNN Accuracy: %.2f%%\n', accuracy);

% Create a mesh grid for visualization
[x1_grid, x2_grid] = meshgrid(linspace(min(X(:,1)), max(X(:,1)), 100), ...
                              linspace(min(X(:,2)), max(X(:,2)), 100));
X_grid = [x1_grid(:), x2_grid(:)];

% Predict on the grid
Y_grid = knn_classify_fast(X_train, Y_train, X_grid, K);
Y_grid = reshape(Y_grid, size(x1_grid));

% Plot decision boundary
figure; hold on;
scatter(X_train(Y_train == 1, 1), X_train(Y_train == 1, 2), 'bo');
scatter(X_train(Y_train == 0, 1), X_train(Y_train == 0, 2), 'bo');
xlabel('Feature 1'); ylabel('Feature 2');
title('KNN Clustering');
legend('Clusters','Location','SE');
grid on;
hold off;

save_all_figs_OPTION('results/knn1','png',1)

% Plot decision boundary
figure; hold on;
scatter(X_train(Y_train == 1, 1), X_train(Y_train == 1, 2), 'bo');
scatter(X_train(Y_train == 0, 1), X_train(Y_train == 0, 2), 'ro');
% contourf(x1_grid, x2_grid, Y_grid, 'LineColor', 'none', 'FaceAlpha', 0.1);
contourf(x1_grid, x2_grid, Y_grid, 'LineWidth', 0.8, 'FaceAlpha', 0.1);
colormap([1 0.8 0.8; 0.8 0.8 1]);
xlabel('Feature 1'); ylabel('Feature 2');
title('KNN Clustering - Decision Boundary');
legend('Class 1', 'Class 0','Location','SE');
grid on;
hold off;

save_all_figs_OPTION('results/knn2','png',1)

function Y_pred = knn_classify_fast(X_train, Y_train, X_test, K)
    % Optimized KNN using matrix operations
    num_test = size(X_test, 1);
    Y_pred = zeros(num_test, 1);

    for i = 1:num_test
        % Compute Euclidean distances using matrix operations
        distances = vecnorm(X_train - X_test(i, :), 2, 2);

        % Sort and select K nearest neighbors
        [~, idx] = sort(distances, 'ascend');
        nearest_labels = Y_train(idx(1:K));

        % Predict majority label
        Y_pred(i) = mode(nearest_labels);
    end
end

function Y_pred = knn_classify(X_train, Y_train, X_test, K)
    % Inputs:
    %   X_train - Training features (NxM)
    %   Y_train - Training labels (Nx1)
    %   X_test - Test features (PxM)
    %   K - Number of neighbors
    % Output:
    %   Y_pred - Predicted labels for test set (Px1)

    num_test = size(X_test, 1);
    Y_pred = zeros(num_test, 1);

    for i = 1:num_test
        % Compute distances
        distances = sqrt(sum((X_train - X_test(i, :)).^2, 2));

        % Find the K nearest neighbors
        [~, sorted_idx] = sort(distances, 'ascend');
        nearest_labels = Y_train(sorted_idx(1:K));

        % Predict the most common class
        Y_pred(i) = mode(nearest_labels);
    end
end



