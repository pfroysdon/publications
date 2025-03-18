#!/usr/bin/env python3
"""
cnn_tutorial.py
---------------
This tutorial demonstrates a simple CNN on the “cameraman” image.
It applies a convolution (using a fixed kernel), ReLU activation, max pooling,
and a fully connected layer.
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import data, img_as_float

def my_conv2(I, K):
    H, W = I.shape
    kH, kW = K.shape
    outH = H - kH + 1
    outW = W - kW + 1
    S = np.zeros((outH, outW))
    for i in range(outH):
        for j in range(outW):
            patch = I[i:i+kH, j:j+kW]
            S[i,j] = np.sum(patch * K)
    return S

def max_pool(A, pool_size):
    H, W = A.shape
    Hp = H // pool_size
    Wp = W // pool_size
    P = np.zeros((Hp, Wp))
    for i in range(Hp):
        for j in range(Wp):
            patch = A[i*pool_size:(i+1)*pool_size, j*pool_size:(j+1)*pool_size]
            P[i,j] = np.max(patch)
    return P

# Load and preprocess image
I = img_as_float(data.camera())
plt.figure()
plt.subplot(1,3,1)
plt.imshow(I, cmap='gray')
plt.title("Original Image")
plt.axis('off')

# Convolution parameters: edge detector kernel
K = np.array([[-1, -1, -1],
              [ 0,  0,  0],
              [ 1,  1,  1]])
S = my_conv2(I, K)
A = np.maximum(0, S)  # ReLU
poolSize = 2
P = max_pool(A, poolSize)

# Fully connected layer (flatten pooled features)
f = P.flatten()
W_fc = 0.01 * np.random.randn(1, f.size)
b_fc = 0
y = W_fc @ f + b_fc

plt.subplot(1,3,2)
plt.imshow(A, cmap='gray')
plt.title("ReLU Activation Output")
plt.axis('off')
plt.subplot(1,3,3)
plt.imshow(P, cmap='gray')
plt.title("Max Pooling Output")
plt.axis('off')
plt.tight_layout()
plt.show()

plt.figure()
plt.subplot(1,2,1)
plt.imshow(I, cmap='gray')
plt.title("Original Image")
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(S, cmap='gray')
plt.title("Convolved Image (Valid)")
plt.colorbar()
plt.axis('off')
plt.tight_layout()
plt.show()

print(f"The fully connected layer output (prediction) is: {y[0]:.4f}")
