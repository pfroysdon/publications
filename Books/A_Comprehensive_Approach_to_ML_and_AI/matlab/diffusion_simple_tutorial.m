close all; clear all; clc; 

% Generate synthetic data (1D Gaussian samples)
rng(42);
X = randn(100, 1); % 100 samples, 1 feature

% Set diffusion parameters
num_steps = 50;
beta = 0.02;

% Apply forward diffusion
X_noisy = forward_diffusion(X, num_steps, beta);

% Define a simple denoising model (Gaussian smoothing)
model = @(x) 0.9 * x; 

% Apply reverse diffusion (denoising)
X_denoised = reverse_diffusion(X_noisy, num_steps, beta, model);

% Plot results
figure; hold on;
plot(X, 'bo', 'DisplayName', 'Original Data');
plot(X_noisy, 'ro', 'DisplayName', 'Noised Data');
plot(X_denoised, 'go', 'DisplayName', 'Denoised Data');
title('Diffusion Process');
legend;
grid on;
hold off;

% save_all_figs_OPTION('results/diffusionSimple','png',1)

function X_noisy = forward_diffusion(X, num_steps, beta)
    % Implements the forward diffusion process
    % Inputs:
    %   X - Original data (NxM)
    %   num_steps - Number of diffusion steps
    %   beta - Noise schedule (scalar)
    % Output:
    %   X_noisy - Noised data at each step

    X_noisy = X;
    for t = 1:num_steps
        noise = sqrt(beta) * randn(size(X));
        X_noisy = sqrt(1 - beta) * X_noisy + noise;
    end
end

function X_denoised = reverse_diffusion(X_noisy, num_steps, beta, model)
    % Implements the reverse diffusion process (denoising)
    % Inputs:
    %   X_noisy - Noised data (NxM)
    %   num_steps - Number of diffusion steps
    %   beta - Noise schedule (scalar)
    %   model - Function handle for denoising model
    % Output:
    %   X_denoised - Reconstructed data

    X_denoised = X_noisy;
    for t = num_steps:-1:1
        predicted_noise = model(X_denoised);
        X_denoised = (X_denoised - sqrt(beta) * predicted_noise) / sqrt(1 - beta);
    end
end


