close all; clear all; clc; 

% Load or generate a dataset (e.g., a Swiss Roll)
[X, t] = generateSwissRoll(1000); % Assume a function that generates Swiss Roll data
Y = myLLE(X, 12, 2);

function Y = myLLE(X, K, d)
    % X: Data matrix of size N x D
    % K: Number of nearest neighbors
    % d: Target dimensionality
    N = size(X,1);
    
    % Compute pairwise distances
    dist = squareform(pdist(X));
    
    % Initialize weight matrix W (N x N)
    W = zeros(N, N);
    
    for i = 1:N
        % Find the indices of K nearest neighbors (excluding self)
        [~, idx] = sort(dist(i,:));
        neighbors = idx(2:K+1);
        
        % Compute the local covariance matrix
        Z = X(i,:) - X(neighbors,:);
        C = Z * Z';
        
        % Regularization (for numerical stability)
        C = C + eye(K)*1e-3*trace(C);
        
        % Solve for weights that minimize the reconstruction error
        w = C \ ones(K, 1);
        w = w / sum(w);
        
        % Assign the weights to the global weight matrix
        W(i, neighbors) = w';
    end
    
    % Compute the matrix M = (I - W)'*(I - W)
    M = (eye(N) - W)' * (eye(N) - W);
    
    % Compute eigenvalues and eigenvectors
    [eigvec, eigval] = eig(M);
    
    % Sort eigenvalues in ascending order
    [~, idx] = sort(diag(eigval));
    eigvec = eigvec(:, idx);
    
    % Discard the first eigenvector (corresponding to zero eigenvalue)
    Y = eigvec(:, 2:d+1);
    
    % Visualization: Plot first two dimensions if d >= 2
    if d >= 2
        figure;
        scatter(Y(:,1), Y(:,2), 25, 'filled');
        title('LLE Embedding (First Two Dimensions)');
        xlabel('Dimension 1'); ylabel('Dimension 2');
        grid on;
    end

    % save_all_figs_OPTION('results/manifoldLearning2','png',1)
end

function [X, t] = generateSwissRoll(N)
% generateSwissRoll Generate a Swiss Roll dataset.
%   [X, t] = generateSwissRoll(N) generates a Swiss Roll with N data points.
%   X is an N-by-3 matrix representing 3D coordinates of the Swiss Roll,
%   and t is the parameter vector that can be used to color or order the points.
%
%   The Swiss Roll is constructed by sampling t uniformly from an interval
%   and computing the corresponding (x, y, z) coordinates as follows:
%       x = t * cos(t)
%       y = h    (random height component)
%       z = t * sin(t)
%
%   Example:
%       [X, t] = generateSwissRoll(1000);
%       scatter3(X(:,1), X(:,2), X(:,3), 20, t, 'filled');
%       title('Swiss Roll Dataset');
%       xlabel('X'); ylabel('Y'); zlabel('Z');
%       colorbar;
%
%   Inputs:
%       N - Number of data points (default: 1000)
%
%   Outputs:
%       X - An N-by-3 matrix containing the Swiss Roll coordinates
%       t - A vector of length N with the underlying parameter values

    if nargin < 1
        N = 1000;  % Default number of points
    end

    % Generate the parameter t uniformly in the interval [1.5*pi, 4.5*pi]
    t = (3*pi/2) * (1 + 2 * rand(N, 1));

    % Generate a height component uniformly in [0, 21]
    h = 21 * rand(N, 1);

    % Compute the 3D coordinates of the Swiss Roll
    X(:,1) = t .* cos(t);  % X-coordinate
    X(:,2) = h;            % Y-coordinate (height)
    X(:,3) = t .* sin(t);  % Z-coordinate

    figure;
    scatter3(X(:,1), X(:,2), X(:,3), 20, t, 'filled');
    view(-17,6) %az=-17,el=6
    title('Swiss Roll Dataset');
    xlabel('X'); ylabel('Y'); zlabel('Z');
    colorbar;
    
    % save_all_figs_OPTION('results/manifoldLearning1','png',1)
end

