% This simple tutorial demonstrates how an autoencoder can learn to 
% compress and reconstruct an image using only basic MATLAB code and 
% gradient descent.
% 
% This script demonstrates a simple autoencoder trained from scratch on
% the built-in "cameraman.tif" image.
%
% Steps:
%   1. Load and downsample the image.
%   2. Flatten the image into a vector.
%   3. Set up a single-hidden-layer autoencoder (encoder + decoder).
%   4. Train the autoencoder using gradient descent to minimize
%      the mean-squared error between the input and its reconstruction.
%   5. Display the original and reconstructed images and plot the loss.

clear; close all; clc;

%% 1. Load and Preprocess the Image
% Load the built-in cameraman image (grayscale)
img = imread('data/cameraman.tif');
img = im2double(img);  % Convert to double precision in [0,1]

% Downsample the image to make the network smaller (e.g., 32x32)
img_ds = imresize(img, [128, 128]); % this is slow
% img_ds = imresize(img, [32, 32]); % this is fast

% Flatten the image into a column vector (each pixel is one feature)
x = img_ds(:);
input_dim = length(x);  % e.g., 32*32 = 1024

%% 2. Set Up the Autoencoder Architecture
hidden_size = 64;  % Number of neurons in the hidden (bottleneck) layer

% Initialize weights and biases (small random values)
rng(1);  % For reproducibility
W1 = 0.01 * randn(hidden_size, input_dim); % Encoder weights
b1 = zeros(hidden_size, 1);                % Encoder bias
W2 = 0.01 * randn(input_dim, hidden_size); % Decoder weights
b2 = zeros(input_dim, 1);                  % Decoder bias

%% 3. Training Parameters
learning_rate = 0.1;
num_epochs = 100;
losses = zeros(num_epochs, 1);

%% 4. Train the Autoencoder using Gradient Descent
for epoch = 1:num_epochs
    % Forward propagation
    z1 = W1 * x + b1;      % Linear combination in hidden layer
    a1 = sigmoid(z1);      % Activation (encoder output)
    z2 = W2 * a1 + b2;     % Linear combination in output layer
    a2 = sigmoid(z2);      % Reconstruction of the input
    
    % Compute the mean squared error loss
    loss = 0.5 * sum((a2 - x).^2);
    losses(epoch) = loss;
    
    % Backpropagation
    % Compute error at the output layer
    delta2 = (a2 - x) .* sigmoid_deriv(z2);
    dW2 = delta2 * a1';    % Gradient for W2
    db2 = delta2;          % Gradient for b2
    
    % Compute error at the hidden layer
    delta1 = (W2' * delta2) .* sigmoid_deriv(z1);
    dW1 = delta1 * x';     % Gradient for W1
    db1 = delta1;          % Gradient for b1
    
    % Update parameters
    W1 = W1 - learning_rate * dW1;
    b1 = b1 - learning_rate * db1;
    W2 = W2 - learning_rate * dW2;
    b2 = b2 - learning_rate * db2;
    
    % Optionally, display progress every 100 epochs
    if mod(epoch, 10) == 0
        fprintf('Epoch %d, Loss: %.4f\n', epoch, loss);
    end
end

%% 5. Reconstruct the Image using the Trained Autoencoder
% Forward propagate the input one last time for the final reconstruction
a2_final = a2;
img_recon = reshape(a2_final, size(img_ds));

%% 6. Visualization
figure('Position', [100, 100, 1200, 500]);

% Display the original downsampled image
subplot(1, 2, 1);
imshow(img_ds);
title('Original Downsampled Image');

% Display the reconstructed image
subplot(1, 2, 2);
imshow(img_recon);
title('Reconstructed Image by Autoencoder');

% save_all_figs_OPTION('results/autoencoder1','png',1)

% Plot the training loss over epochs
figure;
plot(1:num_epochs, losses, 'LineWidth', 2);
xlabel('Epoch');
ylabel('Loss');
title('Training Loss');
grid on;

% save_all_figs_OPTION('results/autoencoder2','png',1)


%% Sigmoid Activation Function
function s = sigmoid(z)
    s = 1 ./ (1 + exp(-z));
end

%% Derivative of the Sigmoid Function
function ds = sigmoid_deriv(z)
    s = sigmoid(z);
    ds = s .* (1 - s);
end
