close all; clear all; clc;

% Generate synthetic data
X = [randn(50,2) + 2; randn(50,2) - 2];  % Two Gaussian clusters

% Set number of clusters
K = 2;

% Run K-means
% [centroids, labels] = k_means(X, K);
[centroids, labels] = k_means_fast(X, K);

% Plot the results
figure; hold on;
gscatter(X(:,1), X(:,2), labels, 'b', 'o'); % Plot data points colored by cluster
title('K-Means Clustering');
xlabel('Feature 1');
ylabel('Feature 2');
legend('Clusters','Location','SE');
grid on;
hold off;

% save_all_figs_OPTION('results/kmeans1','png',1)


% Plot the results
figure; hold on;
gscatter(X(:,1), X(:,2), labels, 'rb', 'o'); % Plot data points colored by cluster
scatter(centroids(:,1), centroids(:,2), 100, 'kX', 'LineWidth', 2); % Plot centroids
title('K-Means Clustering - Centroids Identified');
xlabel('Feature 1');
ylabel('Feature 2');
legend('Cluster 1', 'Cluster 2', 'Centroids','Location','SE');
grid on;
hold off;

% save_all_figs_OPTION('results/kmeans2','png',1)

function [centroids, labels] = k_means_fast(X, K, max_iters)
    % Optimized K-Means Clustering Algorithm in MATLAB
    % Inputs:
    %   X - Data matrix (NxD), where N = number of points, D = feature dimensions
    %   K - Number of clusters
    %   max_iters - Maximum number of iterations (optional, default 100)
    % Outputs:
    %   centroids - Final cluster centroids
    %   labels - Cluster assignments for each point
    
    if nargin < 3
        max_iters = 100;  % Default max iterations
    end
    
    [N, D] = size(X);  % Number of points (N) and feature dimensions (D)
    
    % Step 1: Initialize K centroids randomly
    rng('default');  % Ensure reproducibility
    centroids = X(randperm(N, K), :);  % Select K random points as initial centroids
    
    labels = zeros(N, 1);  % Initialize cluster labels
    prev_centroids = centroids;  % Store previous centroids for convergence check
    
    for iter = 1:max_iters
        % Step 2: Compute distances from all points to all centroids using vectorized operations
        distances = pdist2(X, centroids, 'euclidean');  % NxK distance matrix
        [~, labels] = min(distances, [], 2);  % Assign each point to the nearest centroid
        
        % Step 3: Compute new centroids (mean of assigned points)
        for k = 1:K
            if any(labels == k)
                centroids(k, :) = mean(X(labels == k, :), 1);  % Efficient centroid update
            else
                % Handle empty cluster by reinitializing to a random point
                centroids(k, :) = X(randi(N), :);
            end
        end
        
        % Step 4: Check for convergence
        if max(max(abs(centroids - prev_centroids))) < 1e-6
            break;
        end
        prev_centroids = centroids;
    end
end


function [centroids, labels] = k_means(X, K, max_iters)
    % K-Means Clustering Algorithm in MATLAB
    % Inputs:
    %   X - Data matrix (NxD), where N is the number of points and D is the number of features
    %   K - Number of clusters
    %   max_iters - Maximum number of iterations (optional, default 100)
    % Outputs:
    %   centroids - Final cluster centroids
    %   labels - Cluster assignments for each point
    
    if nargin < 3
        max_iters = 100;  % Default maximum iterations
    end
    
    [N, D] = size(X);  % Number of points (N) and feature dimensions (D)
    
    % Step 1: Randomly initialize K centroids
    rng('default');  % Ensure reproducibility
    centroids = X(randperm(N, K), :);  % Select K random points as initial centroids
    
    labels = zeros(N, 1);  % Initialize cluster labels
    prev_centroids = centroids;  % Store previous centroids for convergence check
    
    % Step 2: Iterate until convergence or max iterations
    for iter = 1:max_iters
        % Step 2.1: Assign each data point to the nearest centroid
        for i = 1:N
            distances = sum((centroids - X(i, :)).^2, 2);  % Compute squared Euclidean distance
            [~, labels(i)] = min(distances);  % Assign the closest centroid
        end
        
        % Step 2.2: Compute new centroids as the mean of assigned points
        for k = 1:K
            if sum(labels == k) == 0
                % Handle empty cluster: reinitialize to a random point
                centroids(k, :) = X(randi(N), :);
            else
                centroids(k, :) = mean(X(labels == k, :), 1);
            end
        end
        
        % Step 2.3: Check for convergence (if centroids do not change significantly)
        if max(max(abs(centroids - prev_centroids))) < 1e-6
            break;
        end
        prev_centroids = centroids;
    end
end

