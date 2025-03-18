import numpy as np
import matplotlib.pyplot as plt

# Plot the standard sigmoid function.
x = np.arange(-8, 8, 0.1)
f = 1 / (1 + np.exp(-x))
plt.figure()
plt.plot(x, f, label='sigmoid(x)')
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Sigmoid Function")
plt.legend()
plt.show()

# Adjusting the weights.
weights = [0.5, 1.0, 2.0]
plt.figure()
for w in weights:
    f_w = 1 / (1 + np.exp(-w * x))
    plt.plot(x, f_w, label=f"w = {w}")
plt.xlabel("x")
plt.ylabel("h_w(x)")
plt.title("Effect of Weight on Sigmoid")
plt.legend()
plt.show()

# Effect of bias.
w = 5.0
biases = [-8.0, 0.0, 8.0]
plt.figure()
for b in biases:
    f_wb = 1 / (1 + np.exp(-(w * x + b)))
    plt.plot(x, f_wb, label=f"b = {b}")
plt.xlabel("x")
plt.ylabel("h_wb(x)")
plt.title("Effect of Bias on Sigmoid")
plt.legend()
plt.show()
