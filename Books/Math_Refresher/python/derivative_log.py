# derivative_log.py
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

# f(x) = log(x) and f'(x) = 1/x; start x at 0.1 to avoid log(0)
x1 = np.arange(0.1, 3.1, 0.1)
fx1 = np.log(x1)
dx1 = 1 / x1

plt.figure()
plt.subplot(1, 2, 1)
plt.plot(x1, fx1, '.', linewidth=2)
plt.plot(x1, fx1, 'b')
plt.xlabel(r'$x$', fontsize=12)
plt.ylabel(r'$f(x)$', fontsize=12)
plt.title(r'$f(x) = \log(x)$', fontsize=12)
plt.subplot(1, 2, 2)
plt.plot(x1, dx1, '.', linewidth=2)
plt.plot(x1, dx1, 'b')
plt.xlabel(r'$x$', fontsize=12)
plt.ylabel(r"$f'(x)$", fontsize=12)
plt.title(r"$f'(x) = \frac{1}{x}$", fontsize=12)
plt.savefig("../figures/derivative_log.pdf")
plt.show()
