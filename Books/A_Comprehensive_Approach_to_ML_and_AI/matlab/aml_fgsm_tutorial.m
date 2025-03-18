% Fast Gradient Sign Method (FGSM) Tutorial in MATLAB
%
% This tutorial demonstrates how to generate adversarial examples using FGSM.
% We use a simple logistic regression classifier defined as:
%
%     f(x) = sigmoid( w * x' + b )
%
% on the built-in "cameraman.tif" image. We assume a true label of 1, compute
% the loss as L = -log(f(x)), and then compute the gradient of the loss with
% respect to the input image x. The adversarial example is generated by:
%
%     x_adv = x + ε * sign(∇_x L)
%
% We then display the original and adversarial images and compare classifier outputs.
%
% All functions (sigmoid, etc.) are implemented from scratch.

clear; clc; close all; rng(1);

%% Load and Preprocess the Image
I = imread('data/cameraman.tif');   % Load image
I = im2double(I);                   % Normalize image to [0, 1]
[H, W] = size(I);
inputDim = H * W;                   % Flattened image dimension
x = I(:)';                          % 1 x (H*W) row vector

%% Define a Simple Logistic Regression Classifier
% For demonstration, we simulate a pre-trained classifier with fixed parameters.
% The classifier is defined as:
%     f(x) = sigmoid( w * x' + b )
% We'll choose w as a small vector and b such that f(x) gives a meaningful output.
w = ones(1, inputDim) * 0.001;  % 1 x inputDim weight vector
b = -0.5;                       % bias (scalar)

% Define the classifier function (returns a probability)
classifier = @(x) sigmoid(w * x' + b);

%% Evaluate the Classifier on the Original Image
y_orig = classifier(x);
fprintf('Classifier output on original image: %.4f\n', y_orig);

%% Compute the Loss and Its Gradient with Respect to the Input
% Assume the true label is 1 (positive class). Then the cross-entropy loss is:
%   L = -log(f(x))
loss = -log(y_orig + 1e-8);

% For logistic regression:
%   f(x) = sigmoid(z) with z = w*x' + b, and dL/dz = f(x) - 1 (since true label = 1)
% Since z = w*x' + b, we have dz/dx = w.
% Therefore, the gradient of the loss with respect to x is:
%   ∇_x L = (f(x) - 1) * w
grad = (y_orig - 1) * w;   % 1 x inputDim

%% Generate the Adversarial Example using FGSM
epsilon = 0.1;  % Perturbation magnitude
x_adv = x + epsilon * sign(grad);  % FGSM update

%% Evaluate the Classifier on the Adversarial Example
y_adv = classifier(x_adv);
fprintf('Classifier output on adversarial image: %.4f\n', y_adv);

%% Reshape and Display the Original and Adversarial Images
I_adv = reshape(x_adv, H, W);  % Reshape adversarial vector into image

figure;
subplot(1,2,1);
imshow(I);
title('Original Image');
subplot(1,2,2);
imshow(I_adv);
title('Adversarial Image (FGSM)');

% save_all_figs_OPTION('results/aml_fgsm','png',1)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Local Function Definitions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function s = sigmoid(x)
    % sigmoid computes the element-wise sigmoid function.
    s = 1 ./ (1 + exp(-x));
end
