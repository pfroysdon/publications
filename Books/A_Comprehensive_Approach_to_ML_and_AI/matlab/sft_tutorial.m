% Supervised Fine-Tuning (SFT) Tutorial in MATLAB
%
% In this tutorial we simulate the process of supervised fine-tuning
% (often used in large language models and other deep networks) in a toy
% setting. We first train a “pre-trained” neural network on a larger synthetic
% dataset for a 2-class classification problem. Then, we assume a new (smaller)
% dataset for a downstream task becomes available. Starting from the pre-trained
% weights, we fine-tune the network on the new dataset using gradient descent.
%
% The network architecture is:
%    Input (2 features) -> Hidden layer (10 neurons, ReLU) -> Output (2 neurons, softmax)
%
% The training loss is the cross-entropy between the network’s predictions and the true labels.
% We compare the decision boundaries before and after fine-tuning.
%
% All functions are written from scratch.

clear; clc; close all; rng(1);

%% 1. Generate Pre-training Dataset (Large Dataset)
N_pre = 500;  % number of samples for pre-training
% Generate two classes as Gaussian clusters in 2D:
% Class 1: centered at (1,1)
X1_pre = randn(2, N_pre/2)*0.5 + repmat([1;1],1,N_pre/2);
% Class 2: centered at (3,3)
X2_pre = randn(2, N_pre/2)*0.5 + repmat([3;3],1,N_pre/2);
X_pre = [X1_pre, X2_pre];  % 2 x N_pre data matrix
labels_pre = [ones(1, N_pre/2), 2*ones(1, N_pre/2)];  % labels 1 and 2

% One-hot encoding for pre-training labels (2 x N_pre)
Y_pre = zeros(2, N_pre);
for i = 1:N_pre
    Y_pre(labels_pre(i), i) = 1;
end

%% 2. Pre-train the Model on Pre-training Data
% Network architecture: 2 -> 10 -> 2
inputDim = 2;
hiddenDim = 10;
outputDim = 2;

% Initialize weights for pre-training (simulate pre-trained model)
W1 = randn(hiddenDim, inputDim) * 0.01;
b1 = zeros(hiddenDim, 1);
W2 = randn(outputDim, hiddenDim) * 0.01;
b2 = zeros(outputDim, 1);

lr_pre = 0.01;
numIter_pre = 3000;

for iter = 1:numIter_pre
    % Forward pass on pre-training data:
    Z1 = W1 * X_pre + repmat(b1, 1, N_pre);  % hidden pre-activation (10 x N_pre)
    H = relu(Z1);                           % hidden activation (10 x N_pre)
    logits = W2 * H + repmat(b2, 1, N_pre);   % logits (2 x N_pre)
    probs = softmax(logits);                % softmax probabilities (2 x N_pre)
    
    % Compute cross-entropy loss:
    loss_pre = -mean(sum(Y_pre .* log(probs + 1e-8), 1));
    
    % Backpropagation:
    d_logits = probs - Y_pre;               % (2 x N_pre)
    grad_W2 = (d_logits * H') / N_pre;        % (2 x 10)
    grad_b2 = mean(d_logits, 2);              % (2 x 1)
    
    d_H = W2' * d_logits;                     % (10 x N_pre)
    d_Z1 = d_H .* reluDerivative(Z1);         % (10 x N_pre)
    grad_W1 = (d_Z1 * X_pre') / N_pre;        % (10 x 2)
    grad_b1 = mean(d_Z1, 2);                  % (10 x 1)
    
    % Parameter updates:
    W2 = W2 - lr_pre * grad_W2;
    b2 = b2 - lr_pre * grad_b2;
    W1 = W1 - lr_pre * grad_W1;
    b1 = b1 - lr_pre * grad_b1;
    
    if mod(iter,500)==0
        fprintf('Pre-training Iteration %d, Loss: %.4f\n', iter, loss_pre);
    end
end

% Save pre-trained weights (simulate pre-trained model)
pretrained.W1 = W1;
pretrained.b1 = b1;
pretrained.W2 = W2;
pretrained.b2 = b2;

%% 3. Generate Fine-Tuning Dataset (Smaller, Downstream Task)
N_ft = 100;  % number of samples for fine-tuning
% Generate new data with a slight shift in means:
% Class 1: centered at (1.5, 1.5)
X1_ft = randn(2, N_ft/2)*0.5 + repmat([1.5;1.5],1,N_ft/2);
% Class 2: centered at (2.5, 2.5)
X2_ft = randn(2, N_ft/2)*0.5 + repmat([2.5;2.5],1,N_ft/2);
X_ft = [X1_ft, X2_ft];  % 2 x N_ft
labels_ft = [ones(1, N_ft/2), 2*ones(1, N_ft/2)];

% One-hot encoding for fine-tuning labels (2 x N_ft)
Y_ft = zeros(2, N_ft);
for i = 1:N_ft
    Y_ft(labels_ft(i), i) = 1;
end

%% 4. Supervised Fine-Tuning (SFT)
% We use the pre-trained model as initialization and fine-tune all parameters on the new dataset.
lr_ft = 0.001;
numIter_ft = 2000;

for iter = 1:numIter_ft
    % Forward pass on fine-tuning data:
    Z1 = W1 * X_ft + repmat(b1, 1, N_ft);
    H = relu(Z1);
    logits = W2 * H + repmat(b2, 1, N_ft);
    probs = softmax(logits);
    
    % Compute cross-entropy loss:
    loss_ft = -mean(sum(Y_ft .* log(probs + 1e-8), 1));
    
    % Backpropagation:
    d_logits = probs - Y_ft;
    grad_W2 = (d_logits * H') / N_ft;
    grad_b2 = mean(d_logits, 2);
    
    d_H = W2' * d_logits;
    d_Z1 = d_H .* reluDerivative(Z1);
    grad_W1 = (d_Z1 * X_ft') / N_ft;
    grad_b1 = mean(d_Z1, 2);
    
    % Update parameters:
    W2 = W2 - lr_ft * grad_W2;
    b2 = b2 - lr_ft * grad_b2;
    W1 = W1 - lr_ft * grad_W1;
    b1 = b1 - lr_ft * grad_b1;
    
    if mod(iter,500)==0
        fprintf('Fine-tuning Iteration %d, Loss: %.4f\n', iter, loss_ft);
    end
end

%% 5. Visualization: Decision Boundary Before and After Fine-Tuning

% Create a grid over the input space.
x_min = min(X_ft(1,:)) - 1;
x_max = max(X_ft(1,:)) + 1;
y_min = min(X_ft(2,:)) - 1;
y_max = max(X_ft(2,:)) + 1;
[xGrid, yGrid] = meshgrid(linspace(x_min, x_max, 100), linspace(y_min, y_max, 100));
gridPoints = [xGrid(:)'; yGrid(:)'];

% Function to compute predictions:
predictFun = @(W1, b1, W2, b2, X) softmax(W2 * relu(W1 * X + repmat(b1, 1, size(X,2))) + repmat(b2, 1, size(X,2)));

% Predictions from pre-trained (before fine-tuning) model:
probs_pre = predictFun(pretrained.W1, pretrained.b1, pretrained.W2, pretrained.b2, gridPoints);
[~, pred_pre] = max(probs_pre, [], 1);
pred_pre_grid = reshape(pred_pre, size(xGrid));

% Predictions from fine-tuned model:
probs_ft = predictFun(W1, b1, W2, b2, gridPoints);
[~, pred_ft] = max(probs_ft, [], 1);
pred_ft_grid = reshape(pred_ft, size(xGrid));

% Plot the fine-tuned decision boundary and data.
figure;
subplot(1,2,1);
imagesc([x_min, x_max], [y_min, y_max], pred_pre_grid);
set(gca, 'YDir', 'normal');
colormap(jet);
colorbar;
hold on;
scatter(X_ft(1,labels_ft==1), X_ft(2,labels_ft==1), 50, 'w', 'filled');
scatter(X_ft(1,labels_ft==2), X_ft(2,labels_ft==2), 50, 'k', 'filled');
title('Pre-trained Model Decision Boundary');
xlabel('Feature 1'); ylabel('Feature 2');

subplot(1,2,2);
imagesc([x_min, x_max], [y_min, y_max], pred_ft_grid);
set(gca, 'YDir', 'normal');
colormap(jet);
colorbar;
hold on;
scatter(X_ft(1,labels_ft==1), X_ft(2,labels_ft==1), 50, 'w', 'filled');
scatter(X_ft(1,labels_ft==2), X_ft(2,labels_ft==2), 50, 'k', 'filled');
title('Fine-tuned Model Decision Boundary');
xlabel('Feature 1'); ylabel('Feature 2');


%% 5. Visualization: Decision Boundary Before and After Fine-Tuning using contourf

% Create a grid over the input space based on the fine-tuning data.
x_min = min(X_ft(1,:)) - 1;
x_max = max(X_ft(1,:)) + 1;
y_min = min(X_ft(2,:)) - 1;
y_max = max(X_ft(2,:)) + 1;
[xGrid, yGrid] = meshgrid(linspace(x_min, x_max, 100), linspace(y_min, y_max, 100));
gridPoints = [xGrid(:)'; yGrid(:)'];  % 2 x numGridPoints

% Define a helper function to compute network predictions.
predictFun = @(W1, b1, W2, b2, X) softmax(W2 * relu(W1 * X + repmat(b1, 1, size(X,2))) + repmat(b2, 1, size(X,2)));

% Compute predictions from the pre-trained model.
probs_pre = predictFun(pretrained.W1, pretrained.b1, pretrained.W2, pretrained.b2, gridPoints);
[~, pred_pre] = max(probs_pre, [], 1);  % 1 x numGridPoints
pred_pre_grid = reshape(pred_pre, size(xGrid));  % Reshape into grid shape

% Compute predictions from the fine-tuned model.
probs_ft = predictFun(W1, b1, W2, b2, gridPoints);
[~, pred_ft] = max(probs_ft, [], 1);
pred_ft_grid = reshape(pred_ft, size(xGrid));

% Plot the decision boundaries with the original fine-tuning data.
figure;
% subplot(1,2,1);
    hold on;
    scatter(X_ft(1,labels_ft==1), X_ft(2,labels_ft==1), 50, 'bo');
    scatter(X_ft(1,labels_ft==2), X_ft(2,labels_ft==2), 50, 'ro');
    contourf(xGrid, yGrid, pred_pre_grid, 'k', 'LineWidth', 1, 'FaceAlpha', 0.2);  % Filled contour plot
    colormap([0.8 0.8 1; 1 0.8 0.8]);
    set(gca, 'YDir', 'normal');
    % colormap(jet); colorbar;
    title('Pre-trained Model Decision Boundary');
    xlabel('Feature 1'); ylabel('Feature 2');
    % legend('Class 1', 'Class 2', 'Decision Boundary');
    hold off;
    save_all_figs_OPTION('results/sft1','png',1)
figure;
% subplot(1,2,2);
    hold on;
    scatter(X_ft(1,labels_ft==1), X_ft(2,labels_ft==1), 50, 'bo');
    scatter(X_ft(1,labels_ft==2), X_ft(2,labels_ft==2), 50, 'ro');
    contourf(xGrid, yGrid, pred_ft_grid, 'k', 'LineWidth', 1, 'FaceAlpha', 0.2);
    colormap([0.8 0.8 1; 1 0.8 0.8]);
    set(gca, 'YDir', 'normal');
    % colormap(jet); colorbar;
    title('Fine-tuned Model Decision Boundary');
    xlabel('Feature 1'); ylabel('Feature 2');
    % legend('Class 1', 'Class 2', 'Decision Boundary');
    hold off;
    save_all_figs_OPTION('results/sft2','png',1)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Local Function Definitions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function y = relu(x)
    y = max(x, 0);
end

function d = reluDerivative(x)
    d = double(x > 0);
end

function y = softmax(x)
    % softmax applied column-wise
    x = x - max(x, [], 1);
    exp_x = exp(x);
    y = exp_x ./ sum(exp_x, 1);
end
