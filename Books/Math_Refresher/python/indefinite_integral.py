# indefinite_integral.py
# Indefinite Integral
#-------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

# f(x) = x^2 - 4
# Its antiderivatives:
# F1(x) = (1/3)*x^3 - 4x
# F2(x) = (1/3)*x^3 - 4x + 1
# F3(x) = (1/3)*x^3 - 4x - 1
x = np.arange(-4, 4.1, 0.1)
fx = np.zeros((len(x),))
Fx1 = np.zeros((len(x),))
Fx2 = np.zeros((len(x),))
Fx3 = np.zeros((len(x),))
for ii in range(len(x)):
    fx[ii] = x[ii]**2 - 4
    Fx1[ii] = (1/3) * x[ii]**3 - 4 * x[ii]
    Fx2[ii] = (1/3) * x[ii]**3 - 4 * x[ii] + 1
    Fx3[ii] = (1/3) * x[ii]**3 - 4 * x[ii] - 1

plt.figure(figsize=(8, 8))

plt.subplot(2, 1, 1)
plt.plot(x, fx, '.', linewidth=2)
plt.plot(x, fx, 'b')
plt.xlabel(r'$$x$$', fontsize=12)
plt.ylabel(r'$$f(x)$$', fontsize=12)
plt.title(r'$$f(x) = x^2 - 4$$', fontsize=12)

plt.subplot(2, 1, 2)
plt.plot(x, Fx1, ':', linewidth=2, label=r'$F_1(x)$')
plt.plot(x, Fx2, '-.', linewidth=2, label=r'$F_2(x)$')
plt.plot(x, Fx3, '--', linewidth=2, label=r'$F_3(x)$')
plt.xlabel(r'$$x$$', fontsize=12)
plt.ylabel(r'$$\int f(x) dx$$', fontsize=12)
plt.title(r'$$\int f(x) dx = \frac{1}{3}x^3 - 4x + \{-1,0,1\}$$', fontsize=12)
plt.legend(fontsize=10)
plt.savefig("../figures/indefinite_integral.pdf")
plt.show()
