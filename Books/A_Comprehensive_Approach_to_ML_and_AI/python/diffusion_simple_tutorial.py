#!/usr/bin/env python3
"""
diffusionTutorial_simple.py
---------------------------
This tutorial implements a simple diffusion process on 1D synthetic Gaussian data.
It applies forward diffusion (adding noise) and then a reverse diffusion (using a simple denoising model).
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
X = np.random.randn(100, 1)  # 100 samples, 1 feature

num_steps = 50
beta = 0.02

def forward_diffusion(X, num_steps, beta):
    X_noisy = X.copy()
    for t in range(num_steps):
        noise = np.sqrt(beta) * np.random.randn(*X.shape)
        X_noisy = np.sqrt(1 - beta) * X_noisy + noise
    return X_noisy

def reverse_diffusion(X_noisy, num_steps, beta, model):
    X_denoised = X_noisy.copy()
    for t in range(num_steps, 0, -1):
        predicted_noise = model(X_denoised)
        X_denoised = (X_denoised - np.sqrt(beta) * predicted_noise) / np.sqrt(1 - beta)
    return X_denoised

# Forward diffusion
X_noisy = forward_diffusion(X, num_steps, beta)

# Simple denoising model: here we use a scaling (for demonstration)
model = lambda x: 0.9 * x
X_denoised = reverse_diffusion(X_noisy, num_steps, beta, model)

plt.figure()
plt.plot(X, 'bo-', label="Original Data")
plt.plot(X_noisy, 'ro-', label="Noisy Data")
plt.plot(X_denoised, 'go-', label="Denoised Data")
plt.title("Diffusion Process")
plt.legend()
plt.grid(True)
plt.show()
