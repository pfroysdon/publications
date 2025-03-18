#!/usr/bin/env python3
"""
aml_tutorial_fgsm.py
--------------------
This tutorial demonstrates the Fast Gradient Sign Method (FGSM) to generate adversarial examples.
We use a simple logistic regression classifier defined as:
    f(x) = sigmoid(w*x' + b)
on the built-in cameraman image from scikit-image.
We compute the loss L = -log(f(x)) (with true label 1) and its gradient with respect to the image,
then generate the adversarial example: x_adv = x + ε * sign(grad).
The original and adversarial images are displayed.
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import data, img_as_float

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def main():
    # Load and preprocess the image
    I = img_as_float(data.camera())
    H, W = I.shape
    input_dim = H * W
    x = I.flatten()[None, :]  # shape (1, input_dim)

    # Define a simple logistic regression classifier with fixed parameters.
    w = np.ones((1, input_dim)) * 0.001  # shape (1, input_dim)
    b = -0.5

    def classifier(x_in):
        # x_in shape: (1, input_dim)
        return sigmoid(np.dot(w, x_in.T) + b).flatten()[0]

    y_orig = classifier(x)
    print(f"Classifier output on original image: {y_orig:.4f}")
    loss = -np.log(y_orig + 1e-8)

    # Compute gradient of loss with respect to x.
    # For logistic regression, grad_x L = (f(x) - 1)*w
    grad = (y_orig - 1) * w  # shape (1, input_dim)

    # Generate adversarial example using FGSM.
    epsilon = 0.1
    x_adv = x + epsilon * np.sign(grad)
    # Clip to [0,1]
    x_adv = np.clip(x_adv, 0, 1)
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
    plt.title("Adversarial Image (FGSM)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
