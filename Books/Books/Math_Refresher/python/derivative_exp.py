# derivative_exp.py
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

# f(x) = e^x and f'(x) = e^x
x1 = np.arange(-2, 2.1, 0.1)
fx1 = np.exp(x1)
dx1 = np.exp(x1)

plt.figure()
plt.subplot(1, 2, 1)
plt.plot(x1, fx1, '.', linewidth=2)
plt.plot(x1, fx1, 'b')
plt.xlabel(r'$x$', fontsize=12)
plt.ylabel(r'$f(x)$', fontsize=12)
plt.title(r'$f(x) = e^x$', fontsize=12)
plt.subplot(1, 2, 2)
plt.plot(x1, dx1, '.', linewidth=2)
plt.plot(x1, dx1, 'b')
plt.xlabel(r'$x$', fontsize=12)
plt.ylabel(r"$f'(x)$", fontsize=12)
plt.title(r"$f'(x) = e^x$", fontsize=12)
plt.savefig("../figures/derivative_exp.pdf")
plt.show()
