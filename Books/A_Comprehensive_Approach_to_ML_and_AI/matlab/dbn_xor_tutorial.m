% DBN Tutorial for XOR-like Classification in MATLAB (from scratch)
%
% In this tutorial we:
%   1. Generate a synthetic XOR-like dataset.
%   2. Pretrain an RBM (Restricted Boltzmann Machine) as the first layer.
%   3. Train a logistic regression classifier on the RBM’s hidden representation.
%   4. Fine-tune the entire DBN (both RBM and logistic layer) with supervised backpropagation.
%   5. Visualize the decision boundary.
%
% All functions (RBM training, transformation, logistic training, DBN fine-tuning, etc.)
% are implemented from scratch.

clear; clc; close all; rng(1);

%% 1. Generate Synthetic XOR-like Data
N = 200;  % total number of samples (must be even)
halfN = N/2;
% For XOR, class 0: points in (0,0) and (1,1); class 1: points in (0,1) and (1,0)
X = zeros(N,2);
y = zeros(N,1);  % labels: 0 or 1

% Class 0: half from around (0,0) and half from around (1,1)
X(1:halfN/2, :) = repmat([0, 0], halfN/2, 1) + 0.1*randn(halfN/2,2);
X(halfN/2+1:halfN, :) = repmat([1, 1], halfN/2, 1) + 0.1*randn(halfN/2,2);
y(1:halfN) = 0;

% Class 1: half from around (0,1) and half from around (1,0)
X(halfN+1:halfN+halfN/2, :) = repmat([0, 1], halfN/2, 1) + 0.1*randn(halfN/2,2);
X(halfN+halfN/2+1:end, :) = repmat([1, 0], halfN/2, 1) + 0.1*randn(halfN/2,2);
y(halfN+1:end) = 1;

% Plot the data
figure;
gscatter(X(:,1), X(:,2), y, 'rb', 'oo');
xlabel('x1'); ylabel('x2'); title('XOR-like Data'); grid on;
% save_all_figs_OPTION('results/dbn1','png',1)

%% 2. Pretrain RBM Layer (Unsupervised)
d_visible = 2;
d_hidden = 12;
lr_rbm = 0.05;
epochs_rbm = 3000;
rbm = rbmTrain(X, d_hidden, lr_rbm, epochs_rbm);

% Get hidden representation from RBM (using sigmoid activation)
H = rbmTransform(rbm, X);  % H is N x d_hidden

%% 3. Train Logistic Regression Classifier on RBM Features
lr_lr = 0.1;
epochs_lr = 3000;
[W_lr, b_lr] = logisticTrain(H, y, lr_lr, epochs_lr);

% Evaluate initial performance (pre-fine-tuning)
y_pred = logisticPredict(W_lr, b_lr, H);
initAcc = mean(round(y_pred) == y) * 100;
fprintf('Pretrained DBN accuracy: %.2f%%\n', initAcc);

%% 4. Fine-Tune the Entire DBN (RBM + Logistic Layer)
epochs_ft = 3000;
lr_ft = 0.001;
[rbm, W_lr, b_lr, lossHistory] = dbnFineTune(X, y, rbm, W_lr, b_lr, lr_ft, epochs_ft);

% Evaluate final performance
H_ft = rbmTransform(rbm, X);
y_pred = logisticPredict(W_lr, b_lr, H_ft);
finalAcc = mean(round(y_pred) == y) * 100;
fprintf('Fine-tuned DBN accuracy: %.2f%%\n', finalAcc);

%% 5. Visualize Decision Boundary
[xGrid, yGrid, gridPred] = decisionBoundary(@(x) dbnPredict(x, rbm, W_lr, b_lr), X);
figure;
gscatter(X(:,1), X(:,2), y, 'rb', 'oo'); hold on;
contour(xGrid, yGrid, gridPred, [0.5 0.5], 'k--', 'LineWidth', 2);
xlabel('x1'); ylabel('x2');
title({'DBN Decision Boundary on XOR-like Data'; ...
    sprintf('Pretrained DBN accuracy: %.2f%%', initAcc); ...
    sprintf('Fine-tuned DBN accuracy: %.2f%%', finalAcc)});
legend('Class 0','Class 1','Decision Boundary','Location','Best');
grid on;
hold off;
% save_all_figs_OPTION('results/dbn2','png',1)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Local Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function rbm = rbmTrain(data, num_hidden, learningRate, epochs)
% rbmTrain trains a Restricted Boltzmann Machine (RBM) using CD-1.
% Inputs:
%   data         - n x d_visible data matrix.
%   num_hidden   - number of hidden units.
%   learningRate - learning rate.
%   epochs       - number of training epochs.
% Output:
%   rbm - structure with fields:
%         W: weights (d_visible x num_hidden)
%         b_visible: visible biases (1 x d_visible)
%         b_hidden: hidden biases (1 x num_hidden)
    [n, d_visible] = size(data);
    rbm.W = 0.01 * randn(d_visible, num_hidden);
    rbm.b_visible = zeros(1, d_visible);
    rbm.b_hidden = zeros(1, num_hidden);
    
    for epoch = 1:epochs
        % Positive phase
        pos_hidden_probs = sigmoid(data * rbm.W + repmat(rbm.b_hidden, n, 1));
        pos_associations = data' * pos_hidden_probs;
        
        % Sample hidden activations (binary) for CD-1
        pos_hidden_states = pos_hidden_probs > rand(size(pos_hidden_probs));
        
        % Negative phase: reconstruct visible units
        neg_visible_probs = sigmoid(pos_hidden_states * rbm.W' + repmat(rbm.b_visible, n, 1));
        neg_hidden_probs = sigmoid(neg_visible_probs * rbm.W + repmat(rbm.b_hidden, n, 1));
        neg_associations = neg_visible_probs' * neg_hidden_probs;
        
        % Update weights and biases
        rbm.W = rbm.W + learningRate * ((pos_associations - neg_associations) / n);
        rbm.b_visible = rbm.b_visible + learningRate * (mean(data - neg_visible_probs, 1));
        rbm.b_hidden = rbm.b_hidden + learningRate * (mean(pos_hidden_probs - neg_hidden_probs, 1));
    end
end

function H = rbmTransform(rbm, data)
% rbmTransform computes the hidden representation for data given a trained RBM.
% Uses sigmoid activation.
    H = sigmoid(data * rbm.W + repmat(rbm.b_hidden, size(data,1), 1));
end

function [W, b] = logisticTrain(H, y, learningRate, epochs)
% logisticTrain trains a binary logistic regression classifier.
% Inputs:
%   H           - n x d feature matrix.
%   y           - n x 1 binary labels (0 or 1).
%   learningRate- learning rate.
%   epochs      - number of training epochs.
% Outputs:
%   W - weight vector (d x 1)
%   b - bias (scalar)
    [n, d] = size(H);
    W = 0.01 * randn(d,1);
    b = 0;
    for epoch = 1:epochs
        scores = H * W + b;        % (n x 1)
        y_pred = sigmoid(scores);  % predicted probabilities
        loss = -mean(y .* log(y_pred + 1e-15) + (1-y) .* log(1-y_pred + 1e-15));
        
        % Gradients (binary cross-entropy)
        dscores = y_pred - y;      % (n x 1)
        gradW = (H' * dscores) / n;
        gradb = mean(dscores);
        
        W = W - learningRate * gradW;
        b = b - learningRate * gradb;
        
        % Optionally display progress every 100 epochs
        if mod(epoch,100)==0 || epoch==epochs
            fprintf('Logistic Epoch %d, Loss: %.4f\n', epoch, loss);
        end
    end
end

function y_pred = logisticPredict(W, b, H)
% logisticPredict computes predicted probabilities from logistic regression.
    scores = H * W + b;
    y_pred = sigmoid(scores);
end

function [rbm, W_lr, b_lr, lossHistory] = dbnFineTune(X, y, rbm, W_lr, b_lr, learningRate, epochs)
% dbnFineTune fine-tunes the entire DBN (RBM + Logistic layer) using supervised backpropagation.
% Inputs:
%   X           - n x d_visible data matrix.
%   y           - n x 1 binary labels (0 or 1).
%   rbm         - trained RBM structure.
%   W_lr, b_lr  - logistic regression parameters.
%   learningRate- learning rate.
%   epochs      - number of fine-tuning epochs.
% Outputs:
%   rbm, W_lr, b_lr - updated parameters.
%   lossHistory - vector of loss values.
    [n, ~] = size(X);
    lossHistory = zeros(epochs,1);
    for epoch = 1:epochs
        % Forward pass through RBM layer
        Z1 = X * rbm.W + repmat(rbm.b_hidden, n, 1);
        H = sigmoid(Z1);  % hidden representation
        
        % Logistic layer
        scores = H * W_lr + b_lr;
        y_pred = sigmoid(scores);
        
        % Compute binary cross-entropy loss
        loss = -mean(y .* log(y_pred+1e-15) + (1-y) .* log(1-y_pred+1e-15));
        lossHistory(epoch) = loss;
        
        % Backpropagation:
        % dLoss/d(scores)
        dscores = (y_pred - y) / n;
        
        % Gradients for logistic layer
        gradW_lr = H' * dscores;
        gradb_lr = sum(dscores);
        
        % Backprop into H
        dH = dscores * W_lr';
        dZ1 = dH .* (H .* (1-H));  % derivative of sigmoid
        
        gradW_rbm = X' * dZ1;
        gradb_rbm = sum(dZ1,1);
        
        % Update parameters
        W_lr = W_lr - learningRate * gradW_lr;
        b_lr = b_lr - learningRate * gradb_lr;
        rbm.W = rbm.W - learningRate * gradW_rbm;
        rbm.b_hidden = rbm.b_hidden - learningRate * gradb_rbm;
        
        if mod(epoch,200)==0 || epoch==epochs
            fprintf('Fine-Tune Epoch %d, Loss: %.4f\n', epoch, loss);
        end
    end
end

function [xGrid, yGrid, gridLabels] = decisionBoundary(netFunc, X)
% decisionBoundary evaluates the network function on a grid over the input space.
% netFunc should be a function handle that accepts a matrix of input points (each row is a sample)
% and returns predicted probability (for class 1). We then threshold at 0.5.
    margin = 0.2;
    x_min = min(X(:,1)) - margin;
    x_max = max(X(:,1)) + margin;
    y_min = min(X(:,2)) - margin;
    y_max = max(X(:,2)) + margin;
    [xGrid, yGrid] = meshgrid(linspace(x_min, x_max, 100), linspace(y_min, y_max, 100));
    gridPoints = [xGrid(:), yGrid(:)];
    preds = netFunc(gridPoints);
    gridLabels = reshape(preds, size(xGrid));
    % For visualization, assign label 0 if probability < 0.5, else 1.
    gridLabels = double(gridLabels >= 0.5);
end

function y_out = dbnPredict(x, rbm, W_lr, b_lr)
% dbnPredict performs a forward pass through the DBN for input x.
% x is an m x d_visible matrix.
    m = size(x,1);
    H = sigmoid(x * rbm.W + repmat(rbm.b_hidden, m, 1));
    scores = H * W_lr + b_lr;
    y_out = sigmoid(scores);
end

function s = sigmoid(x)
    s = 1 ./ (1 + exp(-x));
end

function S = softmaxRows(X)
% softmaxRows applies the softmax function row-wise on X.
    X = X - max(X,[],2);
    X_exp = exp(X);
    S = X_exp ./ sum(X_exp, 2);
end
