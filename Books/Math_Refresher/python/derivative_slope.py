# derivative_slope.py
# Derivative as a slope
#-------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

# Close all figures
plt.close('all')

# f(x) = 2*x and its derivative f'(x) = 2
x1 = np.arange(-2, 2.1, 0.1)
fx1 = np.zeros((len(x1),))
dx1 = np.zeros((len(x1),))
for ii in range(len(x1)):
    fx1[ii] = 2 * x1[ii]
    dx1[ii] = 2

# f(x) = x^3 and its derivative g'(x) = 3*x^2
x2 = np.arange(-2, 2.1, 0.1)
fx2 = np.zeros((len(x2),))
dx2 = np.zeros((len(x2),))
for ii in range(len(x2)):
    fx2[ii] = x2[ii] ** 3
    dx2[ii] = 3 * (x2[ii] ** 2)

# Create a 2x2 subplot to display the functions and their derivatives
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.plot(x1, fx1, '.', linewidth=2)
plt.plot(x1, fx1, 'b')
plt.xlabel(r'$$x$$', fontsize=12)
plt.ylabel(r'$$f(x)$$', fontsize=12)
plt.title(r'$$f(x) = 2x$$', fontsize=12)

plt.subplot(2, 2, 2)
plt.plot(x1, dx1, '.', linewidth=2)
plt.plot(x1, dx1, 'b')
plt.xlabel(r'$$x$$', fontsize=12)
plt.ylabel(r"$$f'(x)$$", fontsize=12)
plt.title(r"$$f'(x) = 2$$", fontsize=12)

plt.subplot(2, 2, 3)
plt.plot(x2, fx2, '.', linewidth=2)
plt.plot(x2, fx2, 'b')
plt.xlabel(r'$$x$$', fontsize=12)
plt.ylabel(r'$$g(x)$$', fontsize=12)
plt.title(r'$$g(x) = x^3$$', fontsize=12)

plt.subplot(2, 2, 4)
plt.plot(x2, dx2, '.', linewidth=2)
plt.plot(x2, dx2, 'b')
plt.xlabel(r'$$x$$', fontsize=12)
plt.ylabel(r"$$g'(x)$$", fontsize=12)
plt.title(r"$$g'(x) = 3x^2$$", fontsize=12)

# Save the figure as a PDF (adjust the path as needed)
plt.savefig("../figures/derivative_slope.pdf")
plt.show()
