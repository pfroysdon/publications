# example_4_6.py
# Exercise 4.6
#-------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

# f(x) = (x^2 + 2x) / (x^2)  [The MATLAB code uses an alternate expression]
# Here we compute: f(x) = x + (2/x)
# Note: Division by zero will produce inf; we let numpy handle this.
x = np.arange(-4, 4.1, 0.1)
# Use np.errstate to ignore divide-by-zero warnings
with np.errstate(divide='ignore', invalid='ignore'):
    fx = x + 2 / x

plt.figure()
plt.plot(x, fx, '.', linewidth=2)
plt.plot(x, fx, 'b')
plt.xlabel(r'$$x$$', fontsize=12)
plt.ylabel(r'$$f(x)$$', fontsize=12)
plt.title(r'$$f(x) = \frac{x^2 + 2x}{x^2}$$', fontsize=12)
# Optionally, you can set the y-limits if needed: plt.ylim([-2, 2])
plt.savefig("../figures/ex_3_6.pdf")
plt.show()
