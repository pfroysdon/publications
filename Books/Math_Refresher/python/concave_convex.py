# concave_convex.py
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

# Define x values
x = np.arange(-4, 4.1, 0.1)
# Compute functions: Concave: -x^2, Convex: x^2
fx1 = -x**2
fx2 = x**2

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(x, fx1, '.', linewidth=2)
plt.plot(x, fx1, 'b')
plt.xlabel(r'$x$', fontsize=12)
plt.ylabel(r'$f(x)$', fontsize=12)
plt.title('Concave', fontsize=12)
plt.subplot(1, 2, 2)
plt.plot(x, fx2, '.', linewidth=2)
plt.plot(x, fx2, 'b')
plt.xlabel(r'$x$', fontsize=12)
plt.ylabel(r'$f(x)$', fontsize=12)
plt.title('Convex', fontsize=12)
plt.savefig("../figures/concave_convex.pdf")
plt.show()
