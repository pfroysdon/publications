% A from-scratch implementation of a simple GAN using the built-in "cameraman.tif" image.
%
% This script loads the "cameraman.tif" image, normalizes it, and flattens it into a
% row vector. It then trains a GAN using one-layer networks for both the generator and 
% discriminator. The generator maps a noise vector (of dimension zDim) to a fake image,
% and the discriminator is trained to distinguish between the real image and generated images.
%
% After training, the generator is used to create new images that are displayed alongside
% the real image.

clear; clc; close all; rng(1);

%% Load and Preprocess the Image
I = imread('data/cameraman.tif');
I = im2double(I);  % Normalize the image to [0, 1]
[H, W] = size(I);
imgSize = [H, W];
inputDim = H * W;  % Flattened image dimension
x_real = I(:)';    % 1 x (H*W) row vector

%% GAN Training Parameters
zDim = 100;         % Dimension of noise vector
epochs = 4000;     % Number of training epochs
alphaD = 0.0005;    % Learning rate for the discriminator
alphaG = 0.0005;    % Learning rate for the generator
batchSize = 32;     % Minibatch size

%% Train the GAN
model = ganTrainCameraman(x_real, inputDim, zDim, epochs, alphaD, alphaG, batchSize, imgSize);

%% Generate New Images using the Trained Generator
numGenSamples = 4;
generatedImages = ganPredict(model, numGenSamples);

%% Visualization: Display the real image and generated images
figure;
subplot(1, numGenSamples+1, 1);
imshow(I);
title('Real Image');

for i = 1:numGenSamples
    subplot(1, numGenSamples+1, i+1);
    imshow(generatedImages(:, :, i));
    title(['Generated ', num2str(i)]);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Local Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function model = ganTrainCameraman(realData, inputDim, zDim, epochs, alphaD, alphaG, batchSize, imgSize)
    % ganTrainCameraman trains a simple GAN using one-layer networks.
    % Inputs:
    %   realData: 1 x inputDim flattened image from cameraman.tif
    %   inputDim: dimension of the flattened image (H*W)
    %   zDim: dimension of noise vector
    %   epochs: number of training epochs
    %   alphaD: learning rate for the discriminator
    %   alphaG: learning rate for the generator
    %   batchSize: minibatch size
    %   imgSize: [H, W] of the image
    % Output:
    %   model: structure containing trained generator and discriminator parameters
    
    % Initialize Discriminator parameters (one-layer network)
    % Discriminator: D(x) = sigmoid(w_D * x' + b_D)
    w_D = randn(1, inputDim) * 0.01;
    b_D = 0;
    
    % Initialize Generator parameters (one-layer network)
    % Generator: G(z) = w_G * z + b_G, mapping noise (zDim x 1) to a flattened image (1 x inputDim)
    w_G = randn(inputDim, zDim) * 0.01;
    b_G = zeros(1, inputDim);
    
    for epoch = 1:epochs
        % --- Discriminator Update ---
        % Create a minibatch of real data by replicating the real image vector.
        realBatch = repmat(realData, batchSize, 1);  % [batchSize x inputDim]
        
        % Sample a minibatch of noise vectors (zDim x batchSize)
        Z = randn(zDim, batchSize);
        % Generate fake data: result is [inputDim x batchSize]
        fakeData = w_G * Z + repmat(b_G', 1, batchSize);  % Correct: size [inputDim x batchSize]
        fakeData = fakeData';  % Convert to [batchSize x inputDim]
        
        % Compute discriminator outputs for real data:
        D_real = sigmoid(w_D * realBatch' + b_D);  % [1 x batchSize]
        D_real = D_real(:);  % [batchSize x 1]
        
        % Compute discriminator outputs for fake data:
        D_fake = sigmoid(w_D * fakeData' + b_D);  % [1 x batchSize]
        D_fake = D_fake(:);  % [batchSize x 1]
        
        % Compute discriminator loss:
        loss_D = -mean(log(D_real + 1e-8) + log(1 - D_fake + 1e-8));
        
        % Compute approximate gradients for discriminator parameters:
        % For real data: gradient of log(D(x)) ~ (1 - D(x))
        grad_wD_real = mean(bsxfun(@times, (1 - D_real), realBatch), 1)';
        grad_bD_real = mean(1 - D_real);
        % For fake data: gradient of log(1 - D(x)) ~ -D(x)
        grad_wD_fake = -mean(bsxfun(@times, D_fake, fakeData), 1)';
        grad_bD_fake = mean(-D_fake);
        
        grad_w_D = (grad_wD_real + grad_wD_fake) / 2;
        grad_b_D = (grad_bD_real + grad_bD_fake) / 2;
        
        % Update discriminator parameters
        w_D = w_D - alphaD * grad_w_D';
        b_D = b_D - alphaD * grad_b_D;
        
        % --- Generator Update ---
        % Sample a new minibatch of noise vectors
        Z = randn(zDim, batchSize);
        fakeData = w_G * Z + repmat(b_G', 1, batchSize);  % [inputDim x batchSize]
        fakeData = fakeData';  % [batchSize x inputDim]
        D_fake = sigmoid(w_D * fakeData' + b_D);
        D_fake = D_fake(:);  % [batchSize x 1]
        
        % Generator loss: L_G = -mean(log(D(fake)))
        loss_G = -mean(log(D_fake + 1e-8));
        
        % Compute gradients for generator parameters (simplified)
        % Approximate the gradient with respect to generator output.
        grad_output = (1 - D_fake);  % [batchSize x 1]
        grad_w_G = zeros(size(w_G));
        for i = 1:batchSize
            grad_w_G = grad_w_G + grad_output(i) * Z(:, i)';
        end
        grad_w_G = grad_w_G / batchSize;
        grad_b_G = mean(grad_output) * ones(1, inputDim);  % Simplified
        
        % Update generator parameters
        w_G = w_G - alphaG * grad_w_G;
        b_G = b_G - alphaG * grad_b_G;
        
        % Optionally, display losses every 1000 epochs
        if mod(epoch, 1000) == 0
            fprintf('Epoch %d, Loss_D: %.4f, Loss_G: %.4f\n', epoch, loss_D, loss_G);
        end
    end
    
    % Save trained parameters in the model structure
    model.w_D = w_D;
    model.b_D = b_D;
    model.w_G = w_G;
    model.b_G = b_G;
    model.noiseDim = zDim;
    model.imgSize = imgSize;
end

function generatedImages = ganPredict(model, numSamples)
    % ganPredict generates images using the trained generator.
    %   numSamples: number of images to generate
    %   generatedImages: output images of size [H, W, numSamples]
    
    z = randn(model.noiseDim, numSamples);  % [noiseDim x numSamples]
    fakeFlat = model.w_G * z + repmat(model.b_G', 1, numSamples);  % [inputDim x numSamples]
    fakeFlat = sigmoid(fakeFlat);  % Scale outputs to [0,1]
    fakeFlat = fakeFlat';  % [numSamples x inputDim]
    
    % Reshape each generated sample to the original image dimensions
    H = model.imgSize(1);
    W = model.imgSize(2);
    generatedImages = reshape(fakeFlat', H, W, numSamples);
end

function s = sigmoid(x)
    s = 1 ./ (1 + exp(-x));
end
