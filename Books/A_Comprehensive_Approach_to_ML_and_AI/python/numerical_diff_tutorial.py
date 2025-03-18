import numpy as np
import matplotlib.pyplot as plt

# Define function and analytical gradient
f = lambda x: np.sum(x**2)
analytical_gradient = lambda x: 2 * x

def numerical_gradient(f, x, h=1e-5):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        e = np.zeros_like(x)
        e[i] = 1
        grad[i] = (f(x + h*e) - f(x - h*e)) / (2*h)
    return grad

# Choose a test point
np.random.seed(1)
x0 = np.random.randn(5)  # 5-dimensional vector

num_grad = numerical_gradient(f, x0)

print("Test point x0:")
print(x0)
print("Analytical gradient:")
print(analytical_gradient(x0))
print("Numerical gradient:")
print(num_grad)
error_norm = np.linalg.norm(num_grad - analytical_gradient(x0))
print(f"Difference (L2 norm) between gradients: {error_norm:.6f}")

# Bar graph comparison
plt.figure()
bar_width = 0.35
indices = np.arange(len(x0))
plt.bar(indices, analytical_gradient(x0), bar_width, label='Analytical')
plt.bar(indices + bar_width, num_grad, bar_width, label='Numerical')
plt.xlabel("Dimension")
plt.ylabel("Gradient Value")
plt.title("Comparison of Analytical and Numerical Gradients")
plt.legend()
plt.show()
