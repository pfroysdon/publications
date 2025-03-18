close all; clear all; clc;  

% Generate sample data from a mixture of two Gaussians
X = [randn(100,1); randn(100,1)+3];
[pi_est, mu1, mu2, sigma1, sigma2, ll] = myEM(X, 100, 1e-4);
figure;
plot(ll, 'LineWidth', 2);
title('Log-Likelihood Convergence');
xlabel('Iteration'); ylabel('Log-Likelihood');

% save_all_figs_OPTION('results/em','png',1)

function [pi_est, mu1, mu2, sigma1, sigma2, loglikelihood] = myEM(X, maxIter, tol)
% myEM Performs the EM algorithm for a 2-component Gaussian mixture model.
%   [pi_est, mu1, mu2, sigma1, sigma2, loglikelihood] = myEM(X, maxIter, tol)
%   Input:
%       X       - A column vector of data points.
%       maxIter - Maximum number of iterations.
%       tol     - Tolerance for convergence of the log-likelihood.
%   Output:
%       pi_est        - Estimated mixing coefficient for component 1.
%       mu1, mu2      - Estimated means of the two Gaussian components.
%       sigma1, sigma2- Estimated standard deviations of the components.
%       loglikelihood - Log-likelihood history.
%
% Example:
%   X = [randn(100,1); randn(100,1)+3];
%   [pi_est, mu1, mu2, sigma1, sigma2, ll] = myEM(X, 100, 1e-4);
%   figure;
%   plot(ll, 'LineWidth', 2);
%   title('Log-Likelihood Convergence');
%   xlabel('Iteration'); ylabel('Log-Likelihood');

    % Initialization
    n = length(X);
    % Random initialization of parameters
    pi_est = 0.5;
    mu1 = mean(X) - std(X)/2;
    mu2 = mean(X) + std(X)/2;
    sigma1 = std(X);
    sigma2 = std(X);
    
    loglikelihood = zeros(maxIter,1);
    
    for iter = 1:maxIter
        % E-step: compute responsibilities for each data point
        gamma = zeros(n, 1);
        for i = 1:n
            p1 = pi_est * normpdf(X(i), mu1, sigma1);
            p2 = (1 - pi_est) * normpdf(X(i), mu2, sigma2);
            gamma(i) = p1 / (p1 + p2);
        end
        
        % M-step: update parameters using the responsibilities
        pi_est_new = mean(gamma);
        mu1_new = sum(gamma .* X) / sum(gamma);
        mu2_new = sum((1 - gamma) .* X) / sum(1 - gamma);
        sigma1_new = sqrt(sum(gamma .* (X - mu1_new).^2) / sum(gamma));
        sigma2_new = sqrt(sum((1 - gamma) .* (X - mu2_new).^2) / sum(1 - gamma));
        
        % Compute log-likelihood
        ll = 0;
        for i = 1:n
            ll = ll + log( pi_est_new * normpdf(X(i), mu1_new, sigma1_new) + ...
                (1 - pi_est_new) * normpdf(X(i), mu2_new, sigma2_new) );
        end
        loglikelihood(iter) = ll;
        
        % Check for convergence
        if iter > 1 && abs(loglikelihood(iter) - loglikelihood(iter-1)) < tol
            break;
        end
        
        % Update parameters
        pi_est = pi_est_new;
        mu1 = mu1_new;
        mu2 = mu2_new;
        sigma1 = sigma1_new;
        sigma2 = sigma2_new;
    end
    
    % Trim loglikelihood history to actual number of iterations
    loglikelihood = loglikelihood(1:iter);
end
