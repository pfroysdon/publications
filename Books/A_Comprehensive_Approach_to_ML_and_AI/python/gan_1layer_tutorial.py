#!/usr/bin/env python3
"""
ganTutorial_1layer.py
---------------------
A simple from-scratch GAN using one-layer networks for both generator and discriminator.
The generator maps a noise vector (zDim) to a fake image; the discriminator distinguishes real from fake.
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import data, img_as_float, transform

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def gan_train_cameraman(real_data, input_dim, z_dim, epochs, alpha_D, alpha_G, batch_size, img_size):
    # Initialize discriminator parameters: one-layer network: D(x)=sigmoid(w_D * x + b_D)
    w_D = 0.01 * np.random.randn(1, input_dim)
    b_D = 0.0
    # Initialize generator parameters: G(z)=w_G * z + b_G
    w_G = 0.01 * np.random.randn(input_dim, z_dim)
    b_G = np.zeros((1, input_dim))
    
    for epoch in range(epochs):
        # For simplicity, we use the same real_data for every batch.
        real_batch = np.tile(real_data, (batch_size, 1))
        Z = np.random.randn(z_dim, batch_size)
        fake_data = (w_G @ Z + b_G.T).T  # shape: (batch_size, input_dim)
        
        # Discriminator outputs
        D_real = sigmoid(np.dot(real_batch, w_D.T) + b_D)
        D_fake = sigmoid(np.dot(fake_data, w_D.T) + b_D)
        
        loss_D = -np.mean(np.log(D_real+1e-8) + np.log(1-D_fake+1e-8))
        
        # Gradients for discriminator (simple approximations)
        grad_wD = np.mean((1 - D_real) * real_batch, axis=0, keepdims=True) - \
                  np.mean(D_fake * fake_data, axis=0, keepdims=True)
        grad_bD = np.mean(1 - D_real) - np.mean(D_fake)
        
        w_D = w_D - alpha_D * grad_wD
        b_D = b_D - alpha_D * grad_bD
        
        # Generator update: aim to maximize log(D(G(z)))
        Z = np.random.randn(z_dim, batch_size)
        fake_data = (w_G @ Z + b_G.T).T
        D_fake = sigmoid(np.dot(fake_data, w_D.T) + b_D)
        loss_G = -np.mean(np.log(D_fake+1e-8))
        
        grad_wG = np.zeros_like(w_G)
        for i in range(batch_size):
            grad_wG += np.outer(fake_data[i], (1 - D_fake[i])).T @ Z[:, i][:,None]
        grad_wG /= batch_size
        grad_bG = np.mean(1 - D_fake) * np.ones((1, input_dim))
        
        w_G = w_G - alpha_G * grad_wG
        b_G = b_G - alpha_G * grad_bG
        
        if (epoch+1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss_D: {loss_D:.4f} | Loss_G: {loss_G:.4f}")
    return {'w_D': w_D, 'b_D': b_D, 'w_G': w_G, 'b_G': b_G, 'z_dim': z_dim, 'img_size': img_size}

def gan_predict(model, num_samples):
    z = np.random.randn(model['z_dim'], num_samples)
    fake_data = (model['w_G'] @ z + model['b_G'].T).T
    # Reshape each row to image shape
    images = np.array([fake_data[i].reshape(model['img_size']) for i in range(num_samples)])
    images = np.clip(images, 0, 1)
    return images

# Main
I = img_as_float(data.camera())
I = transform.resize(I, (64, 64), anti_aliasing=True)
H, W = I.shape
input_dim = H * W
x_real = I.flatten()[None, :]  # shape: (1, input_dim)

# GAN training parameters
z_dim = 100
epochs = 1000
alpha_D = 0.0001
alpha_G = 0.0001
batch_size = 32
img_size = (H, W)

model = gan_train_cameraman(x_real, input_dim, z_dim, epochs, alpha_D, alpha_G, batch_size, img_size)

# Generate new images
num_gen_samples = 4
generated_images = gan_predict(model, num_gen_samples)

# Visualization
plt.figure(figsize=(10,3))
plt.subplot(1, num_gen_samples+1, 1)
plt.imshow(I, cmap='gray')
plt.title("Real Image")
plt.axis('off')
for i in range(num_gen_samples):
    plt.subplot(1, num_gen_samples+1, i+2)
    plt.imshow(generated_images[i], cmap='gray')
    plt.title(f"Gen {i+1}")
    plt.axis('off')
plt.tight_layout()
plt.show()
