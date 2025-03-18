% Projected Gradient Descent (PGD) Adversarial Attack Tutorial in MATLAB
%
% This tutorial demonstrates how to generate adversarial examples using the 
% Projected Gradient Descent (PGD) method. We use a simple logistic regression 
% classifier defined as:
%
%    f(x) = sigmoid( w * x' + b )
%
% on the built-in "cameraman.tif" image. Assuming the true label is 1, we define
% the loss L = -log(f(x)) and compute the gradient of L with respect to the input
% image x (flattened into a row vector). The PGD attack iteratively updates x by:
%
%    x_adv = clip( x_adv + alpha * sign(∇_x L), x - ε, x + ε )
%
% and then clips the result to [0,1].
%
% The original and adversarial images are then displayed.
%
% All functions (sigmoid, numericalGradient, PGD attack, etc.) are implemented from scratch.

clear; clc; close all; rng(1);

%% Load and Preprocess the Image
I = imread('data/cameraman.tif');   % Load image
I = im2double(I);                   % Normalize to [0,1]
[H, W] = size(I);
imgSize = [H, W];
inputDim = H * W;                   % Flattened image dimension
x = I(:)';                          % 1 x inputDim row vector

%% Define a Simple Logistic Regression Classifier
% For demonstration, we define a fixed classifier:
%    f(x) = sigmoid( w * x' + b )
% We'll choose w and b so that the classifier outputs a reasonable probability.
w = ones(1, inputDim) * 0.001;  % 1 x inputDim weight vector
b = -0.5;                       % bias (scalar)
classifier = @(x) sigmoid(w * x' + b);  % function handle that accepts x (1 x inputDim)

%% Evaluate the Classifier on the Original Image
y_orig = classifier(x);
fprintf('Classifier output on original image: %.4f\n', y_orig);

%% PGD Attack Parameters
epsilon = 0.1;   % Maximum ℓ∞ perturbation
alpha = 0.01;    % Step size
numSteps = 10;   % Number of PGD iterations

%% Generate the Adversarial Example using PGD
x_adv = pgdAttack(x, classifier, epsilon, alpha, numSteps);
y_adv = classifier(x_adv);
fprintf('Classifier output on adversarial image: %.4f\n', y_adv);

%% Reshape and Display the Original and Adversarial Images
I_adv = reshape(x_adv, H, W);
figure;
subplot(1,2,1);
imshow(I);
title('Original Image');
subplot(1,2,2);
imshow(I_adv);
title('Adversarial Image (PGD)');

% save_all_figs_OPTION('results/aml_pgd','png',1)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Local Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function x_adv = pgdAttack(x, classifier, epsilon, alpha, numSteps)
    % pgdAttack performs the PGD adversarial attack.
    %
    % Inputs:
    %   x          - Original input (1 x inputDim) row vector.
    %   classifier - Function handle that returns the classifier output (a probability).
    %   epsilon    - Maximum perturbation (ℓ∞ ball radius).
    %   alpha      - Step size.
    %   numSteps   - Number of PGD iterations.
    %
    % Output:
    %   x_adv      - Adversarial example (1 x inputDim) row vector.
    
    x_adv = x;  % Initialize adversarial example as original input.
    for t = 1:numSteps
        t
        % Compute loss: we assume true label = 1, so loss = -log(f(x))
        y_pred = classifier(x_adv);
        loss = -log(y_pred + 1e-8);
        
        % Compute gradient of loss with respect to input x using numerical differentiation.
        grad = numericalGradient(@(x_val) -log(classifier(x_val) + 1e-8), x_adv);
        
        % Update: move in the direction of the sign of the gradient.
        x_adv = x_adv + alpha * sign(grad);
        
        % Project the perturbation: ensure that x_adv is within [x - epsilon, x + epsilon]
        x_adv = max(x_adv, x - epsilon);
        x_adv = min(x_adv, x + epsilon);
        
        % Also clip to [0, 1] since pixel values must remain in that range.
        x_adv = max(x_adv, 0);
        x_adv = min(x_adv, 1);
    end
end

function grad = numericalGradient(f, x)
    % numericalGradient computes the gradient of f at x using central differences.
    %
    % Inputs:
    %   f - Function handle that accepts a row vector and returns a scalar.
    %   x - Row vector at which to compute the gradient.
    %
    % Output:
    %   grad - Numerical gradient (same size as x).
    
    h = 1e-5;
    grad = zeros(size(x));
    for i = 1:length(x)
        e = zeros(size(x));
        e(i) = h;
        grad(i) = (f(x + e) - f(x - e)) / (2*h);
    end
end

function s = sigmoid(x)
    s = 1 ./ (1 + exp(-x));
end
