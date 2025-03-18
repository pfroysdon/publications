#!/usr/bin/env python3
"""
adam_tutorial.py
----------------
This script demonstrates the Adam optimization algorithm by minimizing
the quadratic function:
    f(x) = (x1 - 2)^2 + (x2 + 3)^2
with gradient:
    grad_f(x) = [2*(x1-2), 2*(x2+3)].
It runs 1000 iterations of Adam and plots the loss history.
"""

import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return (x[0] - 2)**2 + (x[1] + 3)**2

def grad_f(x):
    return np.array([2*(x[0]-2), 2*(x[1]+3)])

def main():
    # Adam parameters
    alpha = 0.1
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    num_iterations = 1000

    x = np.array([-5.0, 5.0])
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    loss_history = np.zeros(num_iterations)

    for t in range(1, num_iterations+1):
        g = grad_f(x)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g**2)
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        x = x - alpha * m_hat / (np.sqrt(v_hat) + epsilon)
        loss_history[t-1] = f(x)
        if t % 100 == 0:
            print(f"Iteration {t}: Loss = {loss_history[t-1]:.4f}, x = [{x[0]:.4f}, {x[1]:.4f}]")

    print(f"Optimized solution: x = [{x[0]:.4f}, {x[1]:.4f}]")
    print(f"Final objective value: {f(x):.4f}")

    plt.figure()
    plt.plot(np.arange(1, num_iterations+1), loss_history, linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Objective Function Value")
    plt.title("ADAM Optimization Convergence")
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
