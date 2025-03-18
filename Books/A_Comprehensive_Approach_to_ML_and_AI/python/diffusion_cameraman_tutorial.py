#!/usr/bin/env python3
"""
diffusionTutorial_cameraman.py
------------------------------
A from-scratch implementation of a simple diffusion model applied to the "cameraman" image.
The forward process iteratively adds noise to the image; the reverse process uses the stored noise to reconstruct the image.
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import data, img_as_float, transform

# 1. Load and preprocess image
I = img_as_float(data.camera())
H, W = I.shape
x0 = I.flatten()

# 2. Diffusion model parameters
T = 50  # number of diffusion steps
beta = np.linspace(0.0001, 0.02, T)  # noise schedule
alpha = 1 - beta

x_forward = np.zeros((x0.size, T+1))
x_forward[:, 0] = x0
eps_store = np.zeros((x0.size, T))

# 3. Forward Diffusion Process
for t in range(T):
    eps_t = np.random.randn(x0.size)
    eps_store[:, t] = eps_t
    x_forward[:, t+1] = np.sqrt(alpha[t]) * x_forward[:, t] + np.sqrt(beta[t]) * eps_t

# 4. Reverse Diffusion Process
x_reverse = np.zeros_like(x_forward)
x_reverse[:, T] = x_forward[:, T]
for t in range(T-1, -1, -1):
    x_reverse[:, t] = (x_reverse[:, t+1] - np.sqrt(beta[t]) * eps_store[:, t]) / np.sqrt(alpha[t])

I_recon = x_reverse[:, 0].reshape(H, W)
I_noisy = x_forward[:, T].reshape(H, W)

# 5. Visualization
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.imshow(I, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(I_noisy, cmap='gray')
plt.title("Noisy Image (Forward Process)")
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(I_recon, cmap='gray')
plt.title("Reconstructed Image (Reverse Process)")
plt.axis('off')
plt.show()

plt.figure()
plt.imshow(np.abs(I - I_recon), cmap='gray')
plt.colorbar()
plt.title("Absolute Difference: Original vs. Reconstructed")
plt.show()
