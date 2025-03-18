close all; clear all; clc; 

% Load and Preprocess the Image
I = imread('data/cameraman.tif');
I = im2double(I);           % Normalize image to [0,1]
[H, W] = size(I);
imgSize = [H, W];
x = I(:);                   % Flatten image to vector of size (H*W) x 1

% VAE Hyperparameters
inputDim = numel(x);        % Dimension of the input (H*W)
latentDim = 20;             % Dimension of the latent space
hiddenDim = 100;            % Dimension of the hidden layers
alpha = 0.001;              % Learning rate
epochs = 2000;              % Number of training epochs

% Train the VAE
model = vaeTrain(x, inputDim, latentDim, hiddenDim, alpha, epochs);

% Reconstruct the Image using the Trained VAE
x_recon = vaePredict(model, x);
I_recon = reshape(x_recon, imgSize);

% Visualization
figure;
subplot(1,3,1);
    imshow(I);
    title('Original Image');
subplot(1,3,2);
    imshow(I_recon);
    title('Reconstructed Image');
subplot(1,3,3);
    imagesc(abs(I - I_recon));
    axis square
    colormap gray;
    colorbar;
    title('Absolute Difference');

% save_all_figs_OPTION('results/vae','png',1); 


function model = vaeTrain(x, inputDim, latentDim, hiddenDim, alpha, epochs)
    % vaeTrain trains a simple Variational Autoencoder (VAE) on a single image.
    %   x: Input image vector (inputDim x 1)
    %   inputDim: Dimension of input (H*W)
    %   latentDim: Dimension of latent space
    %   hiddenDim: Dimension of hidden layers
    %   alpha: Learning rate
    %   epochs: Number of training epochs
    %   model: Structure containing trained parameters
    
    % Initialize Encoder parameters
    W_enc = randn(hiddenDim, inputDim) * 0.01;
    b_enc = zeros(hiddenDim, 1);
    W_mu = randn(latentDim, hiddenDim) * 0.01;
    b_mu = zeros(latentDim, 1);
    W_logvar = randn(latentDim, hiddenDim) * 0.01;
    b_logvar = zeros(latentDim, 1);
    
    % Initialize Decoder parameters
    W_dec = randn(hiddenDim, latentDim) * 0.01;
    b_dec = zeros(hiddenDim, 1);
    W_out = randn(inputDim, hiddenDim) * 0.01;
    b_out = zeros(inputDim, 1);
    
    % For simplicity, use a single training sample x (the flattened image)
    for epoch = 1:epochs
        % --- Encoder Forward Pass ---
        h_enc = tanh(W_enc * x + b_enc);
        mu = W_mu * h_enc + b_mu;
        logvar = W_logvar * h_enc + b_logvar;
        sigma = exp(0.5 * logvar);
        
        % --- Reparameterization Trick ---
        epsilon = randn(latentDim, 1);
        z = mu + sigma .* epsilon;
        
        % --- Decoder Forward Pass ---
        h_dec = tanh(W_dec * z + b_dec);
        x_hat = sigmoid(W_out * h_dec + b_out);
        
        % --- Loss Computation ---
        % Reconstruction loss (binary cross-entropy)
        recon_loss = -sum(x .* log(x_hat + 1e-8) + (1 - x) .* log(1 - x_hat + 1e-8));
        % KL divergence loss
        kl_loss = -0.5 * sum(1 + logvar - mu.^2 - exp(logvar));
        loss = recon_loss + kl_loss;
        
        % --- Backpropagation (Simplified) ---
        % NOTE: This example uses very simplified gradient updates.
        % In practice, one should derive and implement the full gradients.
        
        % Gradients for decoder output layer (using MSE approximation for simplicity)
        dL_dxhat = (x_hat - x);  % Approximate gradient for reconstruction loss
        
        grad_W_out = dL_dxhat * h_dec';
        grad_b_out = dL_dxhat;
        
        % Update decoder output layer parameters
        W_out = W_out - alpha * grad_W_out;
        b_out = b_out - alpha * grad_b_out;
        
        % (For brevity, we omit full backpropagation through the entire VAE.)
        % A full implementation would compute gradients for W_dec, b_dec, and the encoder parameters.
        
        % Optionally, display loss every 200 epochs
        if mod(epoch, 200) == 0
            fprintf('Epoch %d, Loss: %.4f (Recon: %.4f, KL: %.4f)\n', epoch, loss, recon_loss, kl_loss);
        end
    end
    
    % Save trained parameters in the model structure
    model.W_enc = W_enc;
    model.b_enc = b_enc;
    model.W_mu = W_mu;
    model.b_mu = b_mu;
    model.W_logvar = W_logvar;
    model.b_logvar = b_logvar;
    model.W_dec = W_dec;
    model.b_dec = b_dec;
    model.W_out = W_out;
    model.b_out = b_out;
end

function x_hat = vaePredict(model, x)
    % vaePredict reconstructs the input x using the trained VAE.
    %   x: Input image vector (inputDim x 1)
    %   x_hat: Reconstructed image vector (inputDim x 1)
    
    h_enc = tanh(model.W_enc * x + model.b_enc);
    mu = model.W_mu * h_enc + model.b_mu;
    % For reconstruction, we use the mean (no sampling)
    z = mu;
    h_dec = tanh(model.W_dec * z + model.b_dec);
    x_hat = sigmoid(model.W_out * h_dec + model.b_out);
end

function s = sigmoid(x)
    s = 1 ./ (1 + exp(-x));
end

