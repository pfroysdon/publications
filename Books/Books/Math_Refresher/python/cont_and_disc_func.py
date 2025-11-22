# cont_and_disc_func.py
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

# f(x) = sqrt(x)
x1 = np.arange(0, 10.1, 0.1)
f1 = np.sqrt(x1)

# f(x) = e^x
x2 = np.arange(-2, 2.1, 0.1)
f2 = np.exp(x2)

# f(x) = 1 + (1/x^2) -- use np.where to avoid division by zero
x3 = np.arange(-4, 4.1, 0.1)
f3 = 1 + np.where(x3 == 0, np.nan, 1/(x3**2))

# f(x) = floor(x)
x4 = np.arange(0, 6, 0.1)
f4 = np.floor(x4)

plt.figure(figsize=(10, 8))
plt.subplot(2, 2, 1)
plt.plot(x1, f1, '.', linewidth=2)
plt.plot(x1, f1, 'b')
plt.xlabel(r'$x$', fontsize=12)
plt.ylabel(r'$f(x)$', fontsize=12)
plt.title(r'$f(x) = \sqrt{x}$', fontsize=12)

plt.subplot(2, 2, 2)
plt.plot(x2, f2, '.', linewidth=2)
plt.plot(x2, f2, 'b')
plt.xlabel(r'$x$', fontsize=12)
plt.ylabel(r'$f(x)$', fontsize=12)
plt.title(r'$f(x) = e^{x}$', fontsize=12)

plt.subplot(2, 2, 3)
plt.plot(x3, f3, '.', linewidth=2)
plt.plot(x3, f3, 'b')
plt.xlabel(r'$x$', fontsize=12)
plt.ylabel(r'$f(x)$', fontsize=12)
plt.title(r'$f(x) = 1 + \frac{1}{x^2}$', fontsize=12)

plt.subplot(2, 2, 4)
plt.plot(x4, f4, '.', linewidth=1)
plt.xlabel(r'$x$', fontsize=12)
plt.ylabel(r'$f(x)$', fontsize=12)
plt.title(r'$f(x) = \lfloor x \rfloor$', fontsize=12)

plt.savefig("../figures/cont_and_disc_func.pdf")
plt.show()
