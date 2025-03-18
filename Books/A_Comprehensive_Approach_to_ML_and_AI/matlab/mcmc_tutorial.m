% Tutorial Overview
% 
%     Parameters:
%         numSamples: total number of MCMC iterations.
%         burnIn: number of initial samples to discard (to allow the chain to converge).
%         proposalStd: standard deviation of the Gaussian proposal distribution.
% 
%     Metropolis-Hastings Sampling:
%         The function metropolisHastings implements the core algorithm:
%             It initializes the chain.
%             Proposes a new state using a Gaussian around the current state.
%             Computes the acceptance probability using the ratio of the target densities.
%             Accepts or rejects the proposal.
%         The target density is provided by the function targetDensity (here, the standard normal).
% 
%     Visualization:
%         After discarding burn-in samples, the script plots a histogram of the MCMC samples normalized as a probability density.
%         It overlays the true standard normal density (using MATLAB’s normpdf) for comparison.
% 
% This self-contained tutorial provides a clear implementation of MCMC via the Metropolis-Hastings algorithm from scratch in MATLAB. Enjoy experimenting and extending this code for other target distributions or proposal mechanisms!
% 
% 
% MCMC Tutorial: Metropolis-Hastings Sampling from a Standard Normal Distribution
%
% This tutorial demonstrates how to implement the Metropolis-Hastings algorithm
% from scratch in MATLAB. Our goal is to sample from the target distribution:
%
%       p(x) ~ exp(-x^2/2)   (a standard normal, up to a constant)
%
% We use a Gaussian proposal distribution with mean equal to the current state.
% The algorithm iterates to build a Markov chain whose stationary distribution is p(x).
%
% After running the sampler, we remove an initial burn-in period and plot the
% histogram of the samples along with the true probability density function.


clear; clc; close all;

%% Parameters
numSamples = 10000;   % Total number of MCMC samples to generate
burnIn = 1000;        % Number of burn-in samples to discard
proposalStd = 1.0;    % Standard deviation of the Gaussian proposal

%% Run Metropolis-Hastings MCMC
samples = metropolisHastings(@targetDensity, numSamples, proposalStd);

% Remove burn-in samples
samples = samples(burnIn+1:end);

%% Visualization: Histogram of Samples vs. True Density
figure;
histogram(samples, 50, 'Normalization', 'pdf', 'EdgeColor', 'none');
hold on;
x = linspace(min(samples)-1, max(samples)+1, 100);
plot(x, normpdf(x, 0, 1), 'r-', 'LineWidth', 2);
title('MCMC Sampling using Metropolis-Hastings');
xlabel('x'); ylabel('Probability Density');
legend('MCMC Samples', 'True Normal PDF');
grid on;
hold off;

% save_all_figs_OPTION('results/mcmc','png',1)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Local Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function samples = metropolisHastings(targetFunc, numSamples, proposalStd)
% metropolisHastings runs the Metropolis-Hastings algorithm.
%
% Inputs:
%   targetFunc  - Function handle to the target density p(x) (up to normalization)
%   numSamples  - Total number of samples to generate
%   proposalStd - Standard deviation of the Gaussian proposal distribution
%
% Output:
%   samples - A column vector (numSamples x 1) of generated samples
%
% The algorithm:
%   1. Initialize the chain with an arbitrary value.
%   2. For each iteration, propose a new state from a Gaussian centered at the current state.
%   3. Accept the proposal with probability:
%          p_accept = min(1, target(proposal) / target(current))
%   4. Otherwise, retain the current state.

    samples = zeros(numSamples, 1);
    % Initialize the chain (e.g., start at 0)
    samples(1) = 0;
    
    for i = 2:numSamples
        current = samples(i-1);
        % Propose a new state: N(current, proposalStd^2)
        proposal = current + proposalStd * randn;
        
        % Compute target densities (unnormalized)
        p_current = targetFunc(current);
        p_proposal = targetFunc(proposal);
        
        % Compute acceptance probability
        acceptance = min(1, p_proposal / p_current);
        
        % Accept or reject the proposal
        if rand < acceptance
            samples(i) = proposal;
        else
            samples(i) = current;
        end
    end
end

function p = targetDensity(x)
% targetDensity returns the unnormalized probability density of a standard normal.
%
% For a standard normal, p(x) = exp(-x^2/2) (ignoring the normalization constant).
    p = exp(-0.5 * x.^2);
end
