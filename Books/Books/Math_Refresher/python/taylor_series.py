# taylor_series.py
# Taylor Series
#-------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

plt.close('all')

# Define the symbolic variable and function
x = sp.symbols('x')
f = sp.sin(x) / x

# Compute Taylor series approximations
# t6: series up to order 6 (i.e. expansion up to x^6)
t6_series = sp.series(f, x, 0, 7).removeO()
# t8: series up to order 8 (expansion up to x^8)
t8_series = sp.series(f, x, 0, 9).removeO()
# t10: series up to order 10 (expansion up to x^10)
t10_series = sp.series(f, x, 0, 11).removeO()

# Print the series approximations
print("Taylor series up to O(x^6):", t6_series)
print("Taylor series up to O(x^8):", t8_series)
print("Taylor series up to O(x^10):", t10_series)

# Convert symbolic expressions to numerical functions for plotting
f_func   = sp.lambdify(x, f, 'numpy')
t6_func  = sp.lambdify(x, t6_series, 'numpy')
t8_func  = sp.lambdify(x, t8_series, 'numpy')
t10_func = sp.lambdify(x, t10_series, 'numpy')

# Create a range of x values for plotting
x_plot = np.linspace(-20, 20, 400)
x_zoom = np.linspace(-6, 6, 400)

# Plot the functions over a wide interval
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(x_plot, f_func(x_plot), 'k', linewidth=2, label=r'$$\frac{\sin(x)}{x}$$')
plt.plot(x_plot, t6_func(x_plot), ':', linewidth=2, label=r'$$\mathcal{O}(x^6)$$ approx.')
plt.plot(x_plot, t8_func(x_plot), '-.', linewidth=2, label=r'$$\mathcal{O}(x^8)$$ approx.')
plt.plot(x_plot, t10_func(x_plot), '--', linewidth=2, label=r'$$\mathcal{O}(x^{10})$$ approx.')
plt.xlim([-20, 20])
plt.xlabel(r'$$x$$', fontsize=12)
plt.ylabel(r'$$y$$', fontsize=12)
plt.title('Taylor Series Expansion', fontsize=12)
plt.legend(loc='lower center', fontsize=10)

# Zoomed plot for better detail
plt.subplot(1, 2, 2)
plt.plot(x_zoom, f_func(x_zoom), 'k', linewidth=2, label=r'$$\frac{\sin(x)}{x}$$')
plt.plot(x_zoom, t6_func(x_zoom), ':', linewidth=2, label=r'$$\mathcal{O}(x^6)$$ approx.')
plt.plot(x_zoom, t8_func(x_zoom), '-.', linewidth=2, label=r'$$\mathcal{O}(x^8)$$ approx.')
plt.plot(x_zoom, t10_func(x_zoom), '--', linewidth=2, label=r'$$\mathcal{O}(x^{10})$$ approx.')
plt.xlim([-6, 6])
plt.ylim([-0.5, 1.5])
plt.xlabel(r'$$x$$', fontsize=12)
plt.ylabel(r'$$y$$', fontsize=12)
plt.title('ZOOM: Taylor Series Expansion', fontsize=12)
plt.legend(loc='upper center', fontsize=10)

plt.gcf().set_size_inches(8, 4)
plt.savefig("../figures/taylor_series.pdf")
plt.show()
