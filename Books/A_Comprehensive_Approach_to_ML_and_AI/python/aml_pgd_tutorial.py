#!/usr/bin/env python3
"""
aml_tutorial_pgd.py
-------------------
This tutorial demonstrates the Projected Gradient Descent (PGD) attack to generate adversarial examples.
We use a simple logistic regression classifier defined as:
    f(x) = sigmoid(w*x' + b)
on the built-in cameraman image from scikit-image.
The attack iteratively updates the image:
    x_adv = clip(x_adv + alpha * sign(grad)), and projects it into the ℓ∞ ball
    around the original image.
The original and adversarial images are then displayed.
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import data, img_as_float

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def numerical_gradient(f, x, h=1e-5):
    """Compute the numerical gradient of function f at x (1D numpy array)."""
    grad = np.zeros_like(x)
    for i in range(len(x)):
        e = np.zeros_like(x)
        e[i] = h
        grad[i] = (f(x + e) - f(x - e)) / (2 * h)
    return grad

def pgd_attack(x, classifier, epsilon, alpha, num_steps):
    """Perform PGD attack on input x (1 x input_dim row vector)."""
    x_adv = x.copy()
    for t in range(num_steps):
        # Compute loss; true label assumed to be 1 so loss = -log(f(x))
        y_pred = classifier(x_adv)
        loss = -np.log(y_pred + 1e-8)
        # Compute numerical gradient with respect to x
        grad = numerical_gradient(lambda z: -np.log(classifier(z) + 1e-8), x_adv.flatten())
        grad = grad.reshape(x_adv.shape)
        # Update in the direction of sign(grad)
        x_adv = x_adv + alpha * np.sign(grad)
        # Project the perturbation: ensure x_adv is within [x - epsilon, x + epsilon]
        x_adv = np.clip(x_adv, x - epsilon, x + epsilon)
        # Also clip to [0, 1]
        x_adv = np.clip(x_adv, 0, 1)
    return x_adv

def main():
    # Load and preprocess the image
    I = img_as_float(data.camera())
    H, W = I.shape
    input_dim = H * W
    x = I.flatten()[None, :]  # shape (1, input_dim)

    # Define the classifier
    w = np.ones((1, input_dim)) * 0.001
    b = -0.5
    def classifier(x_in):
        return sigmoid(np.dot(w, x_in.T) + b).flatten()[0]

    y_orig = classifier(x)
    print(f"Classifier output on original image: {y_orig:.4f}")

    # PGD attack parameters
    epsilon = 0.1
    alpha = 0.01
    num_steps = 10

    x_adv = pgd_attack(x, classifier, epsilon, alpha, num_steps)
    y_adv = classifier(x_adv)
    print(f"Classifier output on adversarial image: {y_adv:.4f}")

    I_adv = x_adv.reshape(H, W)

    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(I, cmap='gray')
    plt.title("Original Image")
    plt.axis("off")
    plt.subplot(1,2,2)
    plt.imshow(I_adv, cmap='gray')
    plt.title("Adversarial Image (PGD)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
