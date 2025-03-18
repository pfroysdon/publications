% Simple DBN demonstration using one RBM layer.
% This example uses a toy dataset and a basic implementation of Contrastive Divergence.
close all; clear all; clc; 

% Parameters
numEpochs = 100;
learningRate = 0.05;
numHidden = 10;

% Generate toy data: 200 samples with 20 features (normalized)
X = rand(200, 20);

% Train RBM as the first layer of the DBN
[W, b_hidden, b_visible] = trainRBM(X, numHidden, numEpochs, learningRate);

% Obtain hidden representations (activations)
H = sigmoid(bsxfun(@plus, X * W, b_hidden));

% Visualize hidden activations using a heatmap
figure;
imagesc(H(1:50, :)); % visualize first 50 samples
colorbar;
title('Hidden Activations from the RBM Layer');
xlabel('Hidden Units');
ylabel('Sample Index');

% Display learned weight matrix as an image
figure;
imagesc(W);
colorbar;
title('Learned Weight Matrix');
xlabel('Hidden Units');
ylabel('Input Features');

function [W, b_hidden, b_visible] = trainRBM(X, numHidden, numEpochs, learningRate)
    % trainRBM trains a Restricted Boltzmann Machine using Contrastive Divergence.
    % Inputs:
    %   X           - Data matrix (n_samples x n_features)
    %   numHidden   - Number of hidden units
    %   numEpochs   - Number of training epochs
    %   learningRate- Learning rate for parameter updates
    % Outputs:
    %   W           - Weight matrix (n_features x numHidden)
    %   b_hidden    - Bias vector for hidden units (1 x numHidden)
    %   b_visible   - Bias vector for visible units (1 x n_features)
    
    [n_samples, n_features] = size(X);
    
    % Initialize parameters
    W = 0.1 * randn(n_features, numHidden);
    b_visible = zeros(1, n_features);
    b_hidden = zeros(1, numHidden);
    
    % Number of Contrastive Divergence steps
    k = 1;
    
    for epoch = 1:numEpochs
        % For each sample (for simplicity, process one sample at a time)
        for i = 1:n_samples
            v0 = X(i, :); % initial visible state
            
            % Compute probabilities for hidden units
            h0_prob = sigmoid(v0 * W + b_hidden);
            h0 = double(h0_prob > rand(1, numHidden));
            
            % Perform k steps of Gibbs sampling
            vk = v0;
            hk = h0;
            for step = 1:k
                % Sample visible units given hidden units
                vk_prob = sigmoid(hk * W' + b_visible);
                vk = double(vk_prob > rand(1, n_features));
                % Sample hidden units given visible units
                hk_prob = sigmoid(vk * W + b_hidden);
                hk = double(hk_prob > rand(1, numHidden));
            end
            
            % Compute gradients
            dW = (v0' * h0_prob) - (vk' * hk_prob);
            db_visible = v0 - vk;
            db_hidden = h0_prob - hk_prob;
            
            % Update parameters
            W = W + learningRate * dW;
            b_visible = b_visible + learningRate * db_visible;
            b_hidden = b_hidden + learningRate * db_hidden;
        end
        if mod(epoch, 10) == 0
            fprintf('Epoch %d complete.\n', epoch);
        end
    end
end

function y = sigmoid(x)
    y = 1 ./ (1 + exp(-x));
end