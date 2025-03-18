close all; clear all; clc; 

% diffusionTutorial - A from-scratch implementation of a simple diffusion model.
% This tutorial uses the built-in "cameraman.tif" image as the input data.

%% Load and Preprocess the Image
I = imread('data/cameraman.tif');
I = im2double(I);           % Normalize image to [0, 1]
[H, W] = size(I);
imgSize = [H, W];
x0 = I(:);                  % Flatten image into a vector (H*W x 1)

%% Diffusion Model Parameters
T = 50;  % Number of diffusion steps
% Define a linear noise schedule: beta increases linearly from 0.0001 to 0.02
beta = linspace(0.0001, 0.02, T);
alpha = 1 - beta;

% Pre-allocate matrices to store forward process and noise
x_forward = zeros(length(x0), T+1);
x_forward(:,1) = x0;
eps_store = zeros(length(x0), T);  % to store noise for each step

%% Forward Diffusion Process
% Iteratively add noise to the image
for t = 1:T
    eps_t = randn(size(x0));
    eps_store(:,t) = eps_t;
    x_forward(:,t+1) = sqrt(alpha(t)) * x_forward(:,t) + sqrt(beta(t)) * eps_t;
end

%% Reverse Diffusion Process
% Use the stored noise to reverse the forward process
x_reverse = zeros(size(x_forward));
x_reverse(:,T+1) = x_forward(:,T+1);  % Start with the final noisy image
for t = T:-1:1
    % Reverse update: x_{t-1} = (x_t - sqrt(beta(t))*eps_t)/sqrt(alpha(t))
    x_reverse(:,t) = (x_reverse(:,t+1) - sqrt(beta(t)) * eps_store(:,t)) / sqrt(alpha(t));
end

% Reshape the final reconstructed vector to image format
I_recon = reshape(x_reverse(:,1), H, W);
I_noisy = reshape(x_forward(:,T+1), H, W);

%% Visualization
figure;
subplot(1,3,1);
    imshow(I);
    title('Original Image');
subplot(1,3,2);
    imshow(I_noisy);
    title({'Noisy Image';'(Forward Process)'});
subplot(1,3,3);
    imshow(I_recon);
    title({'Reconstructed Image';'(Reverse Process)'});

% save_all_figs_OPTION('results/diffusion2','png',1)

% Plot absolute difference between original and reconstructed images
figure;
imagesc(abs(I - I_recon));
colormap gray;
colorbar;
title('Absolute Difference: Original vs. Reconstructed');

% save_all_figs_OPTION('results/diffusion3','png',1)

