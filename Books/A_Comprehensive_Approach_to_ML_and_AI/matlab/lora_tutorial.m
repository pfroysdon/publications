% Low Rank Adaptation (LoRA) Tutorial in MATLAB
%
% In this tutorial, we demonstrate Low Rank Adaptation (LoRA) for adapting a 
% pre-trained linear model to a new regression task.
%
% We assume a pre-trained weight matrix (here, a row vector) W0 for a linear 
% model y = W0 * x. For a new task with target weight W_target, instead of 
% retraining the full model, we adapt it by learning a low-rank update:
%
%         W = W0 + B*A
%
% where B (size: 1 x r) and A (size: r x d) are trainable and r << d.
%
% The training objective is to minimize the mean squared error between the model 
% prediction and the target output on new data:
%
%         Loss = mean( (y - (W0+B*A)*x).^2 )
%
% We update B and A via gradient descent while keeping W0 fixed.

clear; clc; close all; rng(1);

%% Parameters
d = 10;           % input dimension
N = 200;          % number of samples
r = 2;            % rank of adaptation (low rank update)
learningRate = 0.01;
numEpochs = 1000;

%% Generate Data
% Create random input data X (each column is a sample)
X = randn(d, N);
% Define the ideal (target) weight vector W_target (1 x d)
W_target = linspace(1, 2, d);  % for example, a vector from 1 to 2
% Generate outputs with some noise
Y = W_target * X + 0.1 * randn(1, N);

%% Pre-trained Weight (W0)
% Assume we have a pre-trained weight vector that is close but not equal to W_target.
W0 = W_target - 0.5;  % for example, W0 is 0.5 less than W_target

%% LoRA Initialization
% We want to learn a low-rank update such that W0 + B*A approximates W_target.
B = randn(1, r) * 0.01;     % B is 1 x r
A = randn(r, d) * 0.01;     % A is r x d

%% Training Loop (Adaptation using LoRA)
lossHistory = zeros(numEpochs, 1);
for epoch = 1:numEpochs
    % Forward pass: Compute adapted weight W = W0 + B*A
    W = W0 + B * A;  % W is 1 x d
    % Compute predictions: Yhat = W * X  (1 x N)
    Yhat = W * X;
    % Compute mean squared error loss:
    loss = mean((Y - Yhat).^2);
    lossHistory(epoch) = loss;
    
    % Compute gradient of loss with respect to W.
    % Let error = Yhat - Y. Then, dLoss/dW = (2/N) * error * X' (but we can ignore constant factor).
    error = Yhat - Y;  % 1 x N
    gradW = (error * X') / N;  % 1 x d
    
    % Since W = W0 + B*A and W0 is fixed, we have:
    % dW/dB = A  (for each element, gradient is the corresponding row of A)
    % dW/dA = B        (for each element, gradient is the corresponding element of B)
    % So, we update B and A using chain rule:
    gradB = gradW * A';   % (1 x d) * (d x r) = 1 x r
    gradA = B' * gradW;   % (r x 1) * (1 x d) = r x d
    
    % Update parameters:
    B = B - learningRate * gradB;
    A = A - learningRate * gradA;
    
    if mod(epoch, 100) == 0
        fprintf('Epoch %d, Loss: %.4f\n', epoch, loss);
    end
end

%% Plot Training Loss
figure;
plot(1:numEpochs, lossHistory, 'LineWidth', 2);
xlabel('Epoch');
ylabel('Mean Squared Error Loss');
title('LoRA Adaptation Training Loss');
grid on;

%% Compare Adapted Weight to Target
W_adapted = W0 + B*A;
fprintf('Pre-trained weight W0: \n');
disp(W0);
fprintf('Target weight W_target: \n');
disp(W_target);
fprintf('Adapted weight W0+B*A: \n');
disp(W_adapted);

%% Predict on a Test Sample (Optional)
% For example, predict on a new random sample:
x_test = randn(d,1);
y_pred = W_adapted * x_test;
y_true = W_target * x_test;
fprintf('Test sample prediction: %.4f (target: %.4f)\n', y_pred, y_true);


% save_all_figs_OPTION('results/lora','png',1)

