% This script demonstrates how to implement DBSCAN from scratch.
% We generate synthetic 2D data (with several clusters and noise),
% run the DBSCAN algorithm, and visualize the resulting clusters.

clear; clc; close all;

% Generate synthetic 2D data for two classes.
X = [randn(50,2); randn(50,2)+3];
y = [ones(50,1); 2*ones(50,1)];

% Set DBSCAN Parameters
eps = 0.8;    % Neighborhood radius
minPts = 5;   % Minimum number of points to form a dense region

% Perform DBSCAN Clustering
labels = dbscanClustering(X, eps, minPts);


% Visualize Clustering Results
figure;
scatter(X(y==1,1), X(y==1,2), 50, 'bo');
hold on;
scatter(X(y==2,1), X(y==2,2), 50, 'bo');
title(sprintf('DBSCAN Clustering', eps, minPts));
xlabel('Feature 1'); ylabel('Feature 2');
hold off;
legend('Clusters','Location','SE');

% save_all_figs_OPTION('results/dbscan1','png',1)


% Visualize Clustering Results
figure;
scatter(X(y==1,1), X(y==1,2), 50, 'ro');
hold on;
scatter(X(y==2,1), X(y==2,2), 50, 'bo');
scatter(X(:,1), X(:,2), 50, labels, 'filled');
colorbar;
% contourf(X(:,1), X(:,2), labels, 'k', 'LineWidth', 1, 'FaceAlpha', 0.1); % Decision boundary
title(sprintf('DBSCAN Clustering (eps = %.2f, minPts = %d)', eps, minPts));
xlabel('Feature 1'); ylabel('Feature 2');
hold off;
legend('Class 0','Class 1','Group','Location','SE');

% save_all_figs_OPTION('results/dbscan2','png',1)


% Display number of clusters (excluding noise labeled as -1) and noise count
numClusters = numel(unique(labels(labels > 0)));
fprintf('Number of clusters found: %d\n', numClusters);
fprintf('Number of noise points: %d\n', sum(labels == -1));


function labels = dbscanClustering(X, eps, minPts)
% dbscanClustering performs DBSCAN clustering on data X.
%
%   labels = dbscanClustering(X, eps, minPts) returns an n x 1 vector
%   of cluster labels for each data point in X. Noise points are labeled as -1.
%
%   Inputs:
%       X      - n x d data matrix.
%       eps    - Neighborhood radius.
%       minPts - Minimum number of points to form a dense region.
%
%   Output:
%       labels - n x 1 vector of cluster labels.

    n = size(X, 1);
    labels = zeros(n, 1);      % 0 means not yet assigned
    visited = false(n, 1);
    clusterId = 0;
    
    for i = 1:n
        if ~visited(i)
            visited(i) = true;
            Neighbors = regionQuery(X, i, eps);
            if numel(Neighbors) < minPts
                % Mark as noise
                labels(i) = -1;
            else
                clusterId = clusterId + 1;
                labels(i) = clusterId;
                % Expand the cluster
                seedSet = Neighbors;
                k = 1;
                while k <= numel(seedSet)
                    j = seedSet(k);
                    if ~visited(j)
                        visited(j) = true;
                        Neighbors_j = regionQuery(X, j, eps);
                        if numel(Neighbors_j) >= minPts
                            seedSet = [seedSet; Neighbors_j(:)];
                        end
                    end
                    if labels(j) == 0
                        labels(j) = clusterId;
                    end
                    k = k + 1;
                end
            end
        end
    end
end

function Neighbors = regionQuery(X, idx, eps)
% regionQuery returns indices of all points in X within distance eps of X(idx,:).
%
%   Neighbors = regionQuery(X, idx, eps) computes the Euclidean distance
%   between point X(idx,:) and all points in X, and returns the indices
%   of those points whose distance is less than or equal to eps.

    point = X(idx, :);
    distances = sqrt(sum((X - point).^2, 2));
    Neighbors = find(distances <= eps);
end
