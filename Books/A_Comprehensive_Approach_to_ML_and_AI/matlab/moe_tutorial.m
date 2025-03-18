% Mixture of Experts (MoE) Tutorial from Scratch in MATLAB
%
% This tutorial demonstrates how to implement a Mixture of Experts model
% from scratch to solve a regression problem. The model consists of:
%   - Two expert models (linear regressors)
%   - A gating network that computes softmax weights for the experts
%
% Given an input x, each expert produces a prediction f_i(x) and the gating
% network outputs weights g_i(x) (summing to 1). The overall prediction is:
%       y_hat = sum_i g_i(x) * f_i(x)
%
% We train the model using stochastic gradient descent with the mean-squared
% error (MSE) loss.

clear; clc; close all;

% Generate Synthetic Data
% We define a piecewise linear function:
%   For x in [0,2]:   y = 2*x + noise
%   For x in (2,4]:   y = 8 - 2*x + noise
N = 200;
X = linspace(0,4,N)';    % 200 x 1 inputs
noise = 0.5 * randn(N,1);
y = zeros(N,1);
for i = 1:N
    if X(i) <= 2
        y(i) = 2 * X(i) + noise(i);
    else
        y(i) = 8 - 2 * X(i) + noise(i);
    end
end

% Plot the synthetic data
figure;
scatter(X, y, 'b', 'filled');
xlabel('x'); ylabel('y');
title('Synthetic Data for Mixture of Experts Regression');
grid on;
% save_all_figs_OPTION('results/moe1','png',1)

% Train Mixture of Experts Model
K = 2;                % Number of experts
learningRate = 0.01;  % Learning rate for SGD
epochs = 10000;       % Number of training epochs
model = moeTrain(X, y, K, learningRate, epochs);

%% Predict on Training Data
y_pred = moePredict(model, X);

% Plot true vs. predicted outputs
figure;
scatter(X, y, 'b', 'filled'); hold on;
plot(X, y_pred, 'r-', 'LineWidth', 2);
xlabel('x'); ylabel('y');
title('Mixture of Experts: True Data vs. Model Prediction');
legend('True Data','Prediction','Location','Best');
grid on;
% save_all_figs_OPTION('results/moe2','png',1)

% Visualize Experts and Gating Network
% For each input, we can examine the experts' individual predictions and
% the gating network's weights.
expert_preds = zeros(N, K);
gating_weights = zeros(N, K);
for i = 1:N
    x_val = X(i);
    [y_hat, expert_out, gating_out] = moePredictGated(model, x_val);
    expert_preds(i,:) = expert_out;
    gating_weights(i,:) = gating_out;
end

% Plot experts' predictions
figure;
plot(X, expert_preds(:,1), 'g--', 'LineWidth', 2); hold on;
plot(X, expert_preds(:,2), 'm--', 'LineWidth', 2);
xlabel('x'); ylabel('Expert Prediction');
title('Experts'' Predictions');
legend('Expert 1', 'Expert 2','Location','Best');
grid on;
% save_all_figs_OPTION('results/moe3','png',1)

% Plot gating network outputs
figure;
plot(X, gating_weights(:,1), 'g-', 'LineWidth', 2); hold on;
plot(X, gating_weights(:,2), 'm-', 'LineWidth', 2);
xlabel('x'); ylabel('Gating Weight');
title('Gating Network Outputs');
legend('Weight Expert 1', 'Weight Expert 2','Location','Best');
grid on;
% save_all_figs_OPTION('results/moe4','png',1)

% Local Functions
function model = moeTrain(X, y, K, learningRate, epochs)
% moeTrain trains a Mixture of Experts model using stochastic gradient descent.
%
% Inputs:
%   X           - n x d input matrix.
%   y           - n x 1 target vector.
%   K           - Number of experts.
%   learningRate- Learning rate for gradient descent.
%   epochs      - Number of training epochs.
%
% Output:
%   model - A structure with the following fields:
%         .gating.W : K x d weight matrix for gating network.
%         .gating.b : K x 1 bias vector for gating network.
%         .expert.W : K x d weight matrix for experts.
%         .expert.b : K x 1 bias vector for experts.

    [n, d] = size(X);
    
    % Initialize gating network parameters
    model.gating.W = randn(K, d) * 0.01;
    model.gating.b = zeros(K, 1);
    
    % Initialize expert parameters (linear models)
    model.expert.W = randn(K, d) * 0.01;
    model.expert.b = zeros(K, 1);
    
    % Training using stochastic gradient descent (SGD)
    for epoch = 1:epochs
        % Shuffle training data indices
        idx = randperm(n);
        totalLoss = 0;
        for i = 1:n
            x = X(idx(i), :)';       % d x 1 column vector
            target = y(idx(i));      % scalar
            
            % Forward Pass:
            % Gating network: compute s = W_g*x + b_g, then softmax
            s = model.gating.W * x + model.gating.b;  % K x 1
            g = softmax(s);                           % K x 1 gating weights
            
            % Experts: each expert produces f_i = W_e(i,:)*x + b_e(i)
            f = model.expert.W * x + model.expert.b;    % K x 1
            % Overall prediction: weighted sum
            y_hat = g' * f;  % scalar
            
            % Compute squared error loss
            e = y_hat - target;
            loss = 0.5 * e^2;
            totalLoss = totalLoss + loss;
            
            % Backward Pass:
            % Gradients for expert parameters:
            % dL/dW_e(i,:) = e * g(i) * x'
            dW_expert = e * (g * x');  % K x d
            db_expert = e * g;         % K x 1
            
            % Gradients for gating parameters:
            % dL/ds_i = e * g(i) * (f(i) - y_hat)
            d_s = e * (g .* (f - y_hat));  % K x 1
            dW_gating = d_s * x';         % K x d
            db_gating = d_s;              % K x 1
            
            % Parameter updates:
            model.expert.W = model.expert.W - learningRate * dW_expert;
            model.expert.b = model.expert.b - learningRate * db_expert;
            model.gating.W = model.gating.W - learningRate * dW_gating;
            model.gating.b = model.gating.b - learningRate * db_gating;
        end
        
        if mod(epoch, 1000) == 0
            fprintf('Epoch %d, Loss: %.4f\n', epoch, totalLoss/n);
        end
    end
end

function y_pred = moePredict(model, X)
% moePredict predicts outputs for the input matrix X using the trained MoE model.
%
% Inputs:
%   model - Trained mixture of experts model.
%   X     - n x d input matrix.
%
% Output:
%   y_pred - n x 1 vector of predictions.
    [n, ~] = size(X);
    y_pred = zeros(n,1);
    for i = 1:n
        x = X(i,:)';
        s = model.gating.W * x + model.gating.b;
        g = softmax(s);
        f = model.expert.W * x + model.expert.b;
        y_pred(i) = g' * f;
    end
end

function [y_hat, expert_out, gating_out] = moePredictGated(model, x)
% moePredictDetailed computes the detailed output of the MoE model for a single input.
%
% Inputs:
%   model - Trained mixture of experts model.
%   x     - d x 1 input vector.
%
% Outputs:
%   y_hat      - Overall prediction (scalar).
%   expert_out - K x 1 vector of expert predictions.
%   gating_out - K x 1 vector of gating network weights.
    s = model.gating.W * x + model.gating.b;
    gating_out = softmax(s);
    expert_out = model.expert.W * x + model.expert.b;
    y_hat = gating_out' * expert_out;
end

function s = softmax(z)
% softmax computes the softmax of vector z.
    z = z - max(z);  % For numerical stability
    ex = exp(z);
    s = ex / sum(ex);
end
