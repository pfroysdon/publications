#!/usr/bin/env python3
"""
autoencoder_tutorial_cameraman.py
---------------------------------
This tutorial demonstrates how an autoencoder can learn to compress and reconstruct an image.
It is trained from scratch on the built-in “cameraman” image using gradient descent.
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import data, transform, img_as_float

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_deriv(z):
    s = sigmoid(z)
    return s * (1 - s)

# 1. Load and Preprocess the Image
img = img_as_float(data.camera())
img_ds = transform.resize(img, (128, 128), anti_aliasing=True)
x = img_ds.flatten().reshape(-1, 1)  # column vector
input_dim = x.shape[0]

# 2. Set Up the Autoencoder Architecture
hidden_size = 64
np.random.seed(1)
W1 = 0.01 * np.random.randn(hidden_size, input_dim)
b1 = np.zeros((hidden_size, 1))
W2 = 0.01 * np.random.randn(input_dim, hidden_size)
b2 = np.zeros((input_dim, 1))

# 3. Training Parameters
learning_rate = 0.1
num_epochs = 100
losses = np.zeros(num_epochs)

# 4. Train the Autoencoder using Gradient Descent
for epoch in range(num_epochs):
    # Forward propagation
    z1 = W1 @ x + b1      # hidden layer pre-activation
    a1 = sigmoid(z1)      # encoder output
    z2 = W2 @ a1 + b2     # output layer pre-activation
    a2 = sigmoid(z2)      # reconstruction

    loss = 0.5 * np.sum((a2 - x)**2)
    losses[epoch] = loss

    # Backpropagation
    delta2 = (a2 - x) * sigmoid_deriv(z2)
    dW2 = delta2 @ a1.T
    db2 = np.mean(delta2, axis=1, keepdims=True)

    delta1 = (W2.T @ delta2) * sigmoid_deriv(z1)
    dW1 = delta1 @ x.T
    db1 = np.mean(delta1, axis=1, keepdims=True)

    # Update parameters
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")

# 5. Reconstruct the Image using the Trained Autoencoder
a2_final = a2
img_recon = a2_final.reshape(img_ds.shape)

# 6. Visualization
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.imshow(img_ds, cmap='gray')
plt.title("Original Downsampled Image")
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(img_recon, cmap='gray')
plt.title("Reconstructed Image by Autoencoder")
plt.axis('off')
plt.show()

plt.figure()
plt.plot(np.arange(num_epochs), losses, linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid(True)
plt.show()
