#!/usr/bin/env python3
"""
ganTutorial_2layer.py
---------------------
A GAN implementation with two hidden layers for both generator and discriminator.
This example uses the "cameraman" image.
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import data, img_as_float, transform

def sigmoid(x):
    return 1/(1+np.exp(-x))

def leaky_relu(x, alpha=0.2):
    return np.maximum(alpha*x, x)

def generator_forward(z, params):
    # Two-layer generator: z -> FC1 -> ReLU -> FC2 -> tanh
    h1 = leaky_relu(np.dot(params['W1'], z) + params['b1'])
    out = np.tanh(np.dot(params['W2'], h1) + params['b2'])
    return out

def discriminator_forward(x, params):
    # Two-layer discriminator: x -> FC1 -> LeakyReLU -> FC2 -> sigmoid
    h1 = leaky_relu(np.dot(params['W1'], x) + params['b1'])
    out = sigmoid(np.dot(params['W2'], h1) + params['b2'])
    return out

# Load image and preprocess
I = img_as_float(data.camera())
I = transform.resize(I, (64,64), anti_aliasing=True)
input_dim = 64*64
x_real = I.flatten()[None, :]  # shape: (1, input_dim)

# Set network dimensions
z_dim = 100
hidden_dim = 256

# Initialize generator parameters
params_G = {
    'W1': 0.01*np.random.randn(hidden_dim, z_dim),
    'b1': np.zeros((hidden_dim, 1)),
    'W2': 0.01*np.random.randn(input_dim, hidden_dim),
    'b2': np.zeros((input_dim, 1))
}

# Initialize discriminator parameters
params_D = {
    'W1': 0.01*np.random.randn(hidden_dim, input_dim),
    'b1': np.zeros((hidden_dim, 1)),
    'W2': 0.01*np.random.randn(1, hidden_dim),
    'b2': 0.0
}

# (For brevity, training loop using simple gradient descent is omitted.
#  One would alternate between updating discriminator and generator losses.)

# After training, use generator to produce images:
num_samples = 4
z = np.random.randn(z_dim, num_samples)
gen_images = generator_forward(z, params_G)
gen_images = gen_images.reshape((64, 64, num_samples))
gen_images = np.clip((gen_images+1)/2, 0, 1)

plt.figure(figsize=(10,3))
plt.subplot(1, num_samples+1, 1)
plt.imshow(I, cmap='gray')
plt.title("Real")
plt.axis('off')
for i in range(num_samples):
    plt.subplot(1, num_samples+1, i+2)
    plt.imshow(gen_images[:,:,i], cmap='gray')
    plt.title(f"Gen {i+1}")
    plt.axis('off')
plt.tight_layout()
plt.show()
