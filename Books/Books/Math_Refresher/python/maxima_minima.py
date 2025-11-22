# maxima_minima.py
# Maxima and Minima
#-------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

# f(x) = x^2 + 2 and its derivative f'(x) = 2x
x = np.arange(-4, 4.1, 0.1)
fx = np.zeros((len(x),))
dx = np.zeros((len(x),))
for ii in range(len(x)):
    fx[ii] = x[ii]**2 + 2
    dx[ii] = 2 * x[ii]

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(x, fx, '.', linewidth=2)
plt.plot(x, fx, 'b')
plt.xlabel(r'$$x$$', fontsize=12)
plt.ylabel(r'$$f(x)$$', fontsize=12)
plt.title(r'$$f(x) = x^2 + 2$$', fontsize=12)

plt.subplot(1, 2, 2)
plt.plot(x, dx, '.', linewidth=2)
plt.plot(x, dx, 'b')
plt.xlabel(r'$$x$$', fontsize=12)
plt.ylabel(r"$$f'(x)$$", fontsize=12)
plt.title(r"$$f'(x) = 2x$$", fontsize=12)

plt.savefig("../figures/maxima_minima.pdf")
plt.show()
