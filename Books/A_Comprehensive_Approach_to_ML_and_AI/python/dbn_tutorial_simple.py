#!/usr/bin/env python3
"""
dbn_tutorial_simple.py
----------------------
This example demonstrates training a Restricted Boltzmann Machine (RBM) as the first layer of a DBN on a toy dataset.
It uses Contrastive Divergence (CD-1) and then visualizes hidden activations and the weight matrix.
"""

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def train_rbm(X, num_hidden, num_epochs, learning_rate):
    n_samples, n_features = X.shape
    W = 0.1 * np.random.randn(n_features, num_hidden)
    b_visible = np.zeros(n_features)
    b_hidden = np.zeros(num_hidden)
    k = 1  # CD-1
    for epoch in range(num_epochs):
        for i in range(n_samples):
            v0 = X[i, :]
            h0_prob = sigmoid(v0 @ W + b_hidden)
            h0 = (h0_prob > np.random.rand(num_hidden)).astype(float)
            vk = v0.copy()
            hk = h0.copy()
            for _ in range(k):
                vk_prob = sigmoid(hk @ W.T + b_visible)
                vk = (vk_prob > np.random.rand(n_features)).astype(float)
                hk_prob = sigmoid(vk @ W + b_hidden)
                hk = (hk_prob > np.random.rand(num_hidden)).astype(float)
            dW = np.outer(v0, h0_prob) - np.outer(vk, hk_prob)
            db_visible = v0 - vk
            db_hidden = h0_prob - hk_prob
            W += learning_rate * dW
            b_visible += learning_rate * db_visible
            b_hidden += learning_rate * db_hidden
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1} complete.")
    return W, b_hidden, b_visible

# Generate toy data: 200 samples with 20 features
np.random.seed(0)
X = np.random.rand(200, 20)
W, b_hidden, b_visible = train_rbm(X, num_hidden=10, num_epochs=100, learning_rate=0.05)
H = sigmoid(X @ W + b_hidden)

plt.figure()
plt.imshow(H[:50, :], aspect='auto', cmap='viridis')
plt.colorbar()
plt.title("Hidden Activations from the RBM Layer")
plt.xlabel("Hidden Units")
plt.ylabel("Sample Index")
plt.show()

plt.figure()
plt.imshow(W, aspect='auto', cmap='viridis')
plt.colorbar()
plt.title("Learned Weight Matrix")
plt.xlabel("Hidden Units")
plt.ylabel("Input Features")
plt.show()
