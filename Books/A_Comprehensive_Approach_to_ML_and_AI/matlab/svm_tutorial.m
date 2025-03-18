close all; clear all; clc; 

% Generate synthetic dataset
rng(42); % Ensure reproducibility
num_samples = 100;

% Class 1 (Blue)
X1 = randn(num_samples/2, 2) + 2;  % Centered at (2,2)
Y1 = ones(num_samples/2, 1);       % Class +1

% Class 2 (Red)
X2 = randn(num_samples/2, 2) - 2;  % Centered at (-2,-2)
Y2 = -ones(num_samples/2, 1);      % Class -1

% Combine dataset
X = [X1; X2];  % Features (Nx2)
Y = [Y1; Y2];  % Labels (Nx1)

% Shuffle data
idx = randperm(num_samples);
X = X(idx, :);
Y = Y(idx);

% Plot dataset
figure; hold on;
scatter(X(Y == 1, 1), X(Y == 1, 2), 'bo'); % Class +1
scatter(X(Y == -1, 1), X(Y == -1, 2), 'bo'); % Class -1
xlabel('Feature 1'); ylabel('Feature 2');
title('SVM Classification');
legend('Clusters','Location','SE');
grid on;
hold off;

save_all_figs_OPTION('results/svm1','png',1)

% Define SVM parameters
C = 1;          % Regularization parameter
sigma = 0.5;    % RBF kernel width

% Train SVM with Linear Kernel
model_linear = svm_train_kernel(X, Y, C, 'linear', sigma);
disp('Linear Kernel SVM Training Complete.');

% Train SVM with RBF Kernel
model_rbf = svm_train_kernel(X, Y, C, 'rbf', sigma);
disp('RBF Kernel SVM Training Complete.');

% Predict on training data using Linear Kernel
Y_pred_linear = svm_predict_kernel(model_linear, X, Y, X, 'linear', sigma);
accuracy_linear = mean(Y_pred_linear == Y) * 100;
fprintf('Linear SVM Accuracy: %.2f%%\n', accuracy_linear);

% Predict on training data using RBF Kernel
Y_pred_rbf = svm_predict_kernel(model_rbf, X, Y, X, 'rbf', sigma);
accuracy_rbf = mean(Y_pred_rbf == Y) * 100;
fprintf('RBF Kernel SVM Accuracy: %.2f%%\n', accuracy_rbf);

% Plot decision boundary for Linear SVM
plot_decision_boundary(model_linear, X, Y, 'linear', sigma);

% save_all_figs_OPTION('results/svm2','png',1)

% Plot decision boundary for RBF SVM
plot_decision_boundary(model_rbf, X, Y, 'rbf', sigma);

% save_all_figs_OPTION('results/svm','png',1)



function plot_decision_boundary(model, X_train, Y_train, kernel, sigma)
    % Create a mesh grid
    [x1_grid, x2_grid] = meshgrid(linspace(min(X_train(:,1)), max(X_train(:,1)), 100), ...
                                  linspace(min(X_train(:,2)), max(X_train(:,2)), 100));
    X_grid = [x1_grid(:), x2_grid(:)];

    % Predict on the grid
    Y_grid = svm_predict_kernel(model, X_train, Y_train, X_grid, kernel, sigma);
    Y_grid = reshape(Y_grid, size(x1_grid));

    % Plot decision boundary
    figure; hold on;
    scatter(X_train(Y_train == 1, 1), X_train(Y_train == 1, 2), 'bo'); % Class +1
    scatter(X_train(Y_train == -1, 1), X_train(Y_train == -1, 2), 'ro'); % Class -1
    % contourf(x1_grid, x2_grid, Y_grid, 'LineColor', 'none', 'FaceAlpha', 0.1);
    contourf(x1_grid, x2_grid, Y_grid, 'LineWidth', 0.8, 'FaceAlpha', 0.1); % Decision boundary
    colormap([1 0.8 0.8; 0.8 0.8 1]);
    xlabel('Feature 1'); ylabel('Feature 2');
    title(['SVM - Decision Boundary (', kernel, ' Kernel)']);
    legend('Class 0','Class 1','Location','SE');
    grid on;
    hold off;
end

function model = svm_train_kernel(X, Y, C, kernel, sigma)
    N = size(X, 1);
    
    % Compute kernel matrix
    K = zeros(N, N);
    for i = 1:N
        for j = 1:N
            if strcmp(kernel, 'rbf')
                K(i, j) = exp(-norm(X(i, :) - X(j, :))^2 / (2 * sigma^2));
            elseif strcmp(kernel, 'linear')
                K(i, j) = X(i, :) * X(j, :)';
            end
        end
    end
    
    % Solve quadratic programming (QP) using MATLAB's built-in solver
    H = (Y * Y') .* K;
    f = -ones(N, 1);
    Aeq = Y'; beq = 0;
    lb = zeros(N, 1); ub = C * ones(N, 1);
    alpha = quadprog(H, f, [], [], Aeq, beq, lb, ub);
    
    % Compute weights and bias
    W = sum((alpha .* Y) .* X);
    support_vectors = X(alpha > 1e-4, :);
    b = mean(Y(alpha > 1e-4) - support_vectors * W');
    
    model.W = W;
    model.b = b;
    model.alpha = alpha;
end

function Y_pred = svm_predict_kernel(model, X_train, Y_train, X_test, kernel, sigma)
    % Predict using SVM with a given kernel
    N_train = size(X_train, 1);
    N_test = size(X_test, 1);
    K_test = zeros(N_test, N_train);

    for i = 1:N_test
        for j = 1:N_train
            if strcmp(kernel, 'rbf')
                K_test(i, j) = exp(-norm(X_test(i, :) - X_train(j, :))^2 / (2 * sigma^2));
            elseif strcmp(kernel, 'linear')
                K_test(i, j) = X_test(i, :) * X_train(j, :)';
            end
        end
    end

    % Compute predictions
    Y_pred = sign(K_test * (model.alpha .* Y_train) + model.b);
end
