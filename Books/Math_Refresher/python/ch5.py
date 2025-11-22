# ch5.py
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

plt.close('all')

# f(x) = 2x and its derivative 2
x_vals = np.arange(-10, 10.1, 1)
f_x = 2 * x_vals
df_x = 2 * np.ones_like(x_vals)
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(x_vals, f_x, 'r', linewidth=2)
plt.title('f(x) = 2x')
plt.subplot(2, 1, 2)
plt.plot(x_vals, df_x, 'r:', linewidth=2)
plt.title("f'(x) = 2")

# g(x) = x^3 and its derivative 3x^2
x_vals = np.arange(-10, 10.1, 1)
g_x = x_vals**3
dg_x = 3 * x_vals**2
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(x_vals, g_x, 'r', linewidth=2)
plt.title('g(x) = x^3')
plt.subplot(2, 1, 2)
plt.plot(x_vals, dg_x, 'r:', linewidth=2)
plt.title("g'(x) = 3x^2")

# f(x) = e^x and its derivative e^x
x_vals = np.arange(-3, 3.01, 0.01)
f_x = np.exp(x_vals)
df_x = np.exp(x_vals)
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(x_vals, f_x, 'r', linewidth=2)
plt.title('f(x) = e^x')
plt.subplot(2, 1, 2)
plt.plot(x_vals, df_x, 'r:', linewidth=2)
plt.title("f'(x) = e^x")

# f(x) = log(x) and its derivative 1/x (avoid x=0)
x_vals = np.arange(0.1, 3.01, 0.01)
f_x = np.log(x_vals)
df_x = 1 / x_vals
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(x_vals, f_x, 'r', linewidth=2)
plt.title('f(x) = log(x)')
plt.subplot(2, 1, 2)
plt.plot(x_vals, df_x, 'r:', linewidth=2)
plt.title("f'(x) = 1/x")

# Taylor series approximations for f(x) = sin(x)/x using sympy
x = sp.symbols('x')
f = sp.sin(x)/x
t6 = sp.series(f, x, 0, 7)   # up to order 6
t8 = sp.series(f, x, 0, 9)   # up to order 8
t10 = sp.series(f, x, 0, 11) # up to order 10
print("Taylor series up to O(x^6):", t6)
print("Taylor series up to O(x^8):", t8)
print("Taylor series up to O(x^10):", t10)

# Convert series to numerical functions
f_func = sp.lambdify(x, f, 'numpy')
t6_func = sp.lambdify(x, sp.series(f, x, 0, 7).removeO(), 'numpy')
t8_func = sp.lambdify(x, sp.series(f, x, 0, 9).removeO(), 'numpy')
t10_func = sp.lambdify(x, sp.series(f, x, 0, 11).removeO(), 'numpy')

x_plot = np.linspace(-6, 6, 400)
plt.figure()
plt.plot(x_plot, t6_func(x_plot), label='Approx. up to O(x^6)')
plt.plot(x_plot, t8_func(x_plot), label='Approx. up to O(x^8)')
plt.plot(x_plot, t10_func(x_plot), label='Approx. up to O(x^10)')
plt.plot(x_plot, f_func(x_plot), label='sin(x)/x')
plt.xlim([-6, 6])
plt.legend(loc='lower center')
plt.title('Taylor Series Expansion')
plt.show()
