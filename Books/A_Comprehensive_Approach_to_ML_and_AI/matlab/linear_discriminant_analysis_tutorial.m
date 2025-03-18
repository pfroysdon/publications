close all; clear all; clc; 

% Generate synthetic data for two classes
rng(1); % For reproducibility
N = 100;
X1 = randn(N,2) + 2;
X2 = randn(N,2) - 2;
X = [X1; X2];
y = [ones(N,1); 2*ones(N,1)];

[W, Z] = lda(X, y);

% Plot original data and projected data
figure;

% --------- Subplot 1: Original Data ---------
subplot(1,2,1);
scatter(X1(:,1), X1(:,2), 'r', 'filled'); hold on;
scatter(X2(:,1), X2(:,2), 'b', 'filled');
title('Original Data');
xlabel('Feature 1'); ylabel('Feature 2'); grid on;

% Draw a dashed line representing the first LDA direction
% 1) Compute the mean of all data
mu = mean(X,1);  % 1x2

% 2) Take the first column of W as the primary discriminant direction
dirVec = W(:,1);  % 2x1

% 3) Generate a set of t-values to plot the line across a suitable range
t = linspace(-80, 80, 100);

% 4) Parametric equation of the line: point(t) = mu + t*dirVec
linePoints = mu + t' * dirVec';  % 100x2

% 5) Plot the dashed line
plot(linePoints(:,1), linePoints(:,2), 'k--', 'LineWidth', 2);

% --------- Subplot 2: Projected Data ---------
subplot(1,2,2);
% For visualization, project data onto the first discriminant axis
scatter(Z(1:N,1), zeros(N,1), 'r', 'filled'); hold on;
scatter(Z(N+1:end,1), zeros(N,1), 'b', 'filled');
title('Data Projected onto First Discriminant');
xlabel('Projection Value'); grid on;

% save_all_figs_OPTION('results/linear_discriminant_analysis','png',1)

% --------- LDA Function ---------
function [W, projectedData] = lda(X, y)
    % lda - Linear Discriminant Analysis implementation
    % Inputs:
    %   X - n x d data matrix
    %   y - n x 1 vector of class labels
    % Outputs:
    %   W - Projection matrix
    %   projectedData - Data projected onto the new subspace

    [n, d] = size(X);
    classes = unique(y);
    C = length(classes);

    % Compute overall mean
    mu = mean(X, 1);

    % Initialize scatter matrices
    Sw = zeros(d, d);
    Sb = zeros(d, d);

    for i = 1:C
        Xi = X(y == classes(i), :);
        Ni = size(Xi, 1);
        mu_i = mean(Xi, 1);
        
        % Within-class scatter
        Sw = Sw + (Xi - mu_i)' * (Xi - mu_i);
        
        % Between-class scatter
        diff = (mu_i - mu);
        Sb = Sb + Ni * (diff' * diff);
    end

    % Solve the generalized eigenvalue problem
    [V, D] = eig(Sb, Sw);
    
    % Sort eigenvectors by descending eigenvalues
    [~, idx] = sort(diag(D), 'descend');
    W = V(:, idx);
    
    % Project the data
    projectedData = X * W;
end
