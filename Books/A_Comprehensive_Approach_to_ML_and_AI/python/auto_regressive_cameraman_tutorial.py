#!/usr/bin/env python3
"""
auto_regressive_deep_learning_cameraman.py
-------------------------------------------
This tutorial demonstrates an auto‐regressive deep learning model applied to the “cameraman” image.
The model is trained to predict the next pixel intensity given a fixed window of previous pixels.
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import data, transform, img_as_float

# Activation functions
def relu(z):
    return np.maximum(0, z)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_deriv(z):
    s = sigmoid(z)
    return s * (1 - s)

# 1. Load and Preprocess the Image
img = img_as_float(data.camera())
# Downsample image to 64x64 for ease of training
img_ds = transform.resize(img, (64, 64), anti_aliasing=True)
x_full = img_ds.flatten()  # 1D sequence of pixel intensities
N = x_full.size

# 2. Create Training Data (auto-regressive)
window_size = 16
num_samples = N - window_size
X_train = np.zeros((window_size, num_samples))
y_train = np.zeros(num_samples)
for i in range(num_samples):
    X_train[:, i] = x_full[i:i+window_size]
    y_train[i] = x_full[i+window_size]

# 3. Set Up Neural Network Architecture
input_dim = window_size
hidden_dim = 50
output_dim = 1
np.random.seed(1)
W1 = 0.01 * np.random.randn(hidden_dim, input_dim)
b1 = np.zeros((hidden_dim, 1))
W2 = 0.01 * np.random.randn(output_dim, hidden_dim)
b2 = np.zeros((output_dim, 1))

# 4. Training Parameters
learning_rate = 0.01
num_epochs = 5000
m = num_samples
losses = np.zeros(num_epochs)

# 5. Train the Model using Batch Gradient Descent
for epoch in range(num_epochs):
    Z1 = W1 @ X_train + b1  # [hidden_dim x m]
    A1 = relu(Z1)
    Z2 = W2 @ A1 + b2       # [1 x m]
    A2 = sigmoid(Z2)
    loss = 0.5 * np.mean((A2 - y_train)**2)
    losses[epoch] = loss

    # Backpropagation
    dZ2 = (A2 - y_train) * sigmoid_deriv(Z2)
    dW2 = (dZ2 @ A1.T) / m
    db2 = np.mean(dZ2, axis=1, keepdims=True)
    dA1 = W2.T @ dZ2
    dZ1 = dA1 * (Z1 > 0)
    dW1 = (dZ1 @ X_train.T) / m
    db1 = np.mean(dZ1, axis=1, keepdims=True)

    # Update parameters
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    if (epoch+1) % 500 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.6f}")

# 6. Auto-Regressive Generation
generated_seq = np.zeros(N)
generated_seq[:window_size] = x_full[:window_size]
for i in range(window_size, N):
    input_seq = generated_seq[i-window_size:i].reshape(-1,1)  # shape (window_size,1)
    z1 = W1 @ input_seq + b1
    a1 = relu(z1)
    z2 = W2 @ a1 + b2
    a2 = sigmoid(z2)
    generated_seq[i] = a2

gen_img = generated_seq.reshape(img_ds.shape)

# 7. Visualization
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.imshow(img_ds, cmap='gray')
plt.title("Original Downsampled Image")
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(gen_img, cmap='gray')
plt.title("Auto-Regressive Generated Image")
plt.axis('off')
plt.show()

plt.figure()
plt.plot(np.arange(num_epochs), losses, linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid(True)
plt.show()
