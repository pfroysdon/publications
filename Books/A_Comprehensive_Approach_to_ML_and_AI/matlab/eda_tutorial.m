close all; clear all; clc;

% Generate a toy dataset: 200 samples, 5 features
X = randn(200, 5);
k = 2;  % Reduce to 2 dimensions
[Z, W] = simplePCA(X, k);

% Plot the projected data
figure;
scatter(Z(:,1), Z(:,2), 50, 'filled');
title('PCA: First Two Principal Components');
xlabel('Principal Component 1');
ylabel('Principal Component 2');
grid on;

% save_all_figs_OPTION('results/eda','png',1)

function [Z, W] = simplePCA(X, k)
    % simplePCA: Perform PCA on data matrix X to reduce its dimensionality to k.
    % Input:  X (n-by-p data matrix), k (number of principal components)
    % Output: Z (projected data, n-by-k), W (projection matrix, p-by-k)
    
    % Step 1: Center the data
    X_mean = mean(X, 1);
    X_centered = X - X_mean;
    
    % Step 2: Compute the covariance matrix
    S = (1/size(X,1)) * (X_centered' * X_centered);
    
    % Step 3: Eigenvalue decomposition
    [V, D] = eig(S);
    eigenvalues = diag(D);
    
    % Step 4: Sort eigenvectors by decreasing eigenvalues
    [~, idx] = sort(eigenvalues, 'descend');
    V = V(:, idx);
    
    % Step 5: Select the top k eigenvectors
    W = V(:, 1:k);
    
    % Step 6: Project the data
    Z = X_centered * W;
end
