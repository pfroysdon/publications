close all; clear all; clc; 

X = [randn(100,1); randn(100,1)+3];
[x_grid, f_hat] = myKDE(X, 0.5, 200);
figure;
plot(x_grid, f_hat, 'LineWidth', 2);
title('Kernel Density Estimate for a Mixture of Gaussians');
xlabel('x'); ylabel('Estimated Density');

% save_all_figs_OPTION('results/kde','png',1)

function [x_grid, f_hat] = myKDE(X, h, num_points)
% myKDE Kernel Density Estimation from scratch.
%
%   [x_grid, f_hat] = myKDE(X, h, num_points) estimates the density function
%   for the data in vector X using a Gaussian kernel with bandwidth h.
%   num_points specifies the number of points in the evaluation grid.

    if nargin < 3
        num_points = 100;
    end

    % Create a grid of evaluation points
    x_min = min(X) - 3*h;
    x_max = max(X) + 3*h;
    x_grid = linspace(x_min, x_max, num_points);
    
    % Initialize density estimate vector
    f_hat = zeros(size(x_grid));
    
    % Define the Gaussian kernel function
    K = @(u) (1/sqrt(2*pi)) * exp(-0.5*u.^2);
    
    % Compute KDE for each grid point
    n = length(X);
    for i = 1:length(x_grid)
        u = (x_grid(i) - X) / h;
        f_hat(i) = sum(K(u));
    end
    f_hat = f_hat / (n * h);
end
