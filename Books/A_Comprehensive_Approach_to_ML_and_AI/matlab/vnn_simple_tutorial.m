% VNN Tutorial from scratch for binary classification
%
% This script demonstrates how to implement a simple feedforward neural network
% with one hidden layer from scratch to classify 2D data.
%
% The network uses:
%   - ReLU activation in the hidden layer.
%   - tanh activation in the output layer (outputs in [-1, 1]).
%   - Mean Squared Error (MSE) loss and gradient descent for training.
%
% Data: Synthetic data with two classes:
%   Class +1 centered at (2,2) and Class -1 centered at (-2,-2).

clear; close all; clc;

% Generate Synthetic Data
rng(1);  % For reproducibility
N = 100;
% Class +1: centered at (2,2)
X1 = randn(N,2) + 2;
% Class -1: centered at (-2,-2)
X2 = randn(N,2) - 2;
X = [X1; X2];       % Data: 200 x 2
y = [ones(N,1); -ones(N,1)];  % Labels: 200 x 1

% Train a Simple VNN from Scratch
hiddenSize = 10;      % Number of hidden neurons
learningRate = 0.02;  % Learning rate
epochs = 500;       % Number of training epochs
[model, lossHistory] = VNNTrain(X, y, hiddenSize, learningRate, epochs);

% Evaluate the VNN on Training Data
y_pred = VNNPredict(model, X);
accuracy = mean(sign(y_pred) == y) * 100;
fprintf('Training Accuracy: %.2f%%\n', accuracy);

% Visualize Decision Boundary
x1_range = linspace(min(X(:,1))-1, max(X(:,1))+1, 100);
x2_range = linspace(min(X(:,2))-1, max(X(:,2))+1, 100);
[X1_grid, X2_grid] = meshgrid(x1_range, x2_range);
gridPoints = [X1_grid(:), X2_grid(:)];
predictions = VNNPredict(model, gridPoints);
Z = reshape(sign(predictions), size(X1_grid));  % use sign for classification

figure;
gscatter(X(:,1), X(:,2), y, 'rb', 'oo');
hold on;
% contourf(X1_grid, X2_grid, Z, [-1 0 1], 'LineColor','none', 'FaceAlpha', 0.1);
contourf(X1_grid, X2_grid, Z, 'LineWidth', 0.8, 'FaceAlpha', 0.1); % Decision boundary
colormap([1 0.8 0.8; 0.8 0.8 1]);
title('VNN Decision Boundary');
xlabel('Feature 1'); ylabel('Feature 2');
legend('Class 0','Class 1','Location','SE');
hold off;
% save_all_figs_OPTION('results/vnn1','png',1)

% Plot Loss History
figure;
plot(lossHistory, 'LineWidth', 2);
xlabel('Epoch'); ylabel('MSE Loss');
title('Training Loss History');
grid on;
% save_all_figs_OPTION('results/vnn2','png',1)


function [model, lossHistory] = VNNTrain(X, y, hiddenSize, learningRate, epochs)
% VNNTrain trains a simple feedforward neural network with one hidden layer.
%
% Inputs:
%   X           - n x d data matrix (n samples, d features)
%   y           - n x 1 vector of labels in {-1, 1}
%   hiddenSize  - number of hidden neurons
%   learningRate- learning rate for gradient descent
%   epochs      - number of training epochs
%
% Outputs:
%   model       - structure containing trained parameters: W1, b1, W2, b2
%   lossHistory - vector of loss values for each epoch

    [n, d] = size(X);
    
    % Initialize weights and biases with small random values
    W1 = randn(hiddenSize, d) * 0.01;
    b1 = zeros(hiddenSize, 1);
    W2 = randn(1, hiddenSize) * 0.01;
    b2 = 0;
    
    lossHistory = zeros(epochs, 1);
    
    for epoch = 1:epochs
        % Forward Pass:
        % Hidden layer: linear transformation + ReLU activation
        Z1 = W1 * X' + repmat(b1, 1, n);  % size: hiddenSize x n
        A1 = max(0, Z1);                  % ReLU activation
        
        % Output layer: linear transformation + tanh activation for output
        Z2 = W2 * A1 + b2;                % size: 1 x n
        A2 = tanh(Z2);                    % tanh activation (outputs in [-1, 1])
        
        % Compute Mean Squared Error Loss
        loss = 0.5 * mean((A2 - y').^2);
        lossHistory(epoch) = loss;
        
        % Backward Pass:
        % Output layer gradients
        dA2 = (A2 - y') / n;                      % 1 x n
        dZ2 = dA2 .* (1 - A2.^2);                  % tanh derivative
        dW2 = dZ2 * A1';                          % 1 x hiddenSize
        db2 = sum(dZ2, 2);                        % scalar
        
        % Backprop to hidden layer
        dA1 = W2' * dZ2;                          % hiddenSize x n
        % Derivative of ReLU: 1 if Z1 > 0, else 0
        dZ1 = dA1;
        dZ1(Z1 <= 0) = 0;
        dW1 = dZ1 * X;                            % hiddenSize x d
        db1 = sum(dZ1, 2);                        % hiddenSize x 1
        
        % Parameter Updates (Gradient Descent)
        W1 = W1 - learningRate * dW1;
        b1 = b1 - learningRate * db1;
        W2 = W2 - learningRate * dW2;
        b2 = b2 - learningRate * db2;
        
        % Optionally display loss every 1000 epochs
        if mod(epoch, 1000) == 0
            fprintf('Epoch %d, Loss: %.4f\n', epoch, loss);
        end
    end
    
    % Store trained parameters in the model structure
    model.W1 = W1;
    model.b1 = b1;
    model.W2 = W2;
    model.b2 = b2;
end

function y_pred = VNNPredict(model, X)
% VNNPredict predicts outputs for the input data X using the trained VNN model.
%
% Inputs:
%   model - structure containing parameters (W1, b1, W2, b2)
%   X     - n x d data matrix
%
% Output:
%   y_pred - n x 1 vector of predicted outputs (real numbers; use sign() for class labels)
    [n, ~] = size(X);
    % Forward pass: compute hidden layer activation
    Z1 = model.W1 * X' + repmat(model.b1, 1, n);
    A1 = max(0, Z1);   % ReLU activation
    % Output layer: tanh activation
    Z2 = model.W2 * A1 + model.b2;
    A2 = tanh(Z2);
    y_pred = A2';      % Convert to n x 1 vector
end
