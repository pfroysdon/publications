# ch1.py
import numpy as np
import matplotlib.pyplot as plt
from math import factorial, sqrt, floor
import math
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting

plt.close('all')

# ----------------------------
# f(x) = x^2 - 1
# ----------------------------
x = np.arange(-2, 2.1, 0.1)
f_x = x**2 - 1

plt.figure()
plt.subplot(1, 2, 1)
plt.plot(x, f_x, 'b.', linewidth=3)
plt.plot(x, f_x, 'b')
plt.subplot(1, 2, 2)
plt.plot(f_x, x, 'b.', linewidth=3)
plt.plot(f_x, x, 'b')
plt.savefig("../figures/functions.pdf")

# ----------------------------
# f(x) = 0.75*x - 6
# ----------------------------
x = np.arange(-2, 6.5, 0.5)
f_x = 0.75 * x - 6
plt.figure()
plt.plot(x, f_x, 'b.', linewidth=3)
plt.plot(x, f_x, 'b')
plt.plot(0, -6, 'ro', markersize=8)
plt.plot(4, -3, 'ro', markersize=8)
plt.savefig("../figures/simple_line.pdf")

# ----------------------------
# f(x) = sin(x)/x
# ----------------------------
x = np.arange(-6, 6.1, 0.1)
f_x = np.sin(x) / x
f_x[np.isnan(f_x)] = 1  # Handle division by zero at x = 0
plt.figure()
plt.plot(x, f_x, 'b.', linewidth=3)
plt.plot(x, f_x, 'b')
plt.savefig("../figures/graph13.pdf")

# ----------------------------
# Piecewise function:
# f(x) = -x^2+2 for x>=0 and f(x) = x^2-2 for x<0
# ----------------------------
x1 = np.arange(0, 4.1, 0.1)
f_x1 = -x1**2 + 2
x2 = np.arange(-4, 0.1, 0.1)
f_x2 = x2**2 - 2
plt.figure()
plt.plot(f_x1, x1, 'b.', markersize=8, linewidth=3, label='x>=0')
plt.plot(f_x2, x2, 'b.', markersize=8, linewidth=3, label='x<0')
plt.plot(f_x1, x1, 'b', linewidth=1)
plt.plot(f_x2, x2, 'b', linewidth=1)
plt.plot(2, 0, 'b*', markersize=10)
plt.plot(-2, 0, 'b*', markersize=10)
plt.savefig("../figures/graph15.pdf")

# ----------------------------
# f(x) = -3*x + 2
# ----------------------------
x = np.arange(-3, 3.1, 0.1)
f_x = -3 * x + 2
plt.figure()
plt.plot(x, f_x, 'b.', linewidth=3)
plt.plot(x, f_x, 'b')
plt.savefig("../figures/ex1_20_1.pdf")

# ----------------------------
# f(x) = sqrt(2)*x - 3
# ----------------------------
x = np.arange(-3, 3.1, 0.1)
f_x = np.sqrt(2) * x - 3
plt.figure()
plt.plot(x, f_x, 'b.', linewidth=3)
plt.plot(x, f_x, 'b')
plt.savefig("../figures/ex1_20_2.pdf")

# ----------------------------
# Summation "by hand"
# ----------------------------
n = 10
x_vec = np.arange(1, n+1)
x_i = x_vec[0]
for ii in range(1, n):
    x_i += x_vec[ii]
result_1 = x_i
result_2 = np.sum(x_vec)
print("Summation by hand:", result_1, "Built-in sum:", result_2)

# ----------------------------
# Summation with a constant "by hand"
# ----------------------------
n = 10
c = 2
x_vec = np.arange(1, n+1)
x_i = c * x_vec[0]
for ii in range(1, n):
    x_i += c * x_vec[ii]
result_1 = x_i

x_vec = np.arange(1, n+1)
x_i = x_vec[0]
for ii in range(1, n):
    x_i += x_vec[ii]
result_2 = c * x_i
print("Summation with constant by hand:", result_1, "and", result_2)

# ----------------------------
# Products "by hand"
# ----------------------------
n = 10
x_vec = np.arange(1, n+1)
x_i = x_vec[0]
for ii in range(1, n):
    x_i *= x_vec[ii]
result_1 = x_i
result_2 = np.prod(x_vec)
print("Product by hand:", result_1, "Built-in prod:", result_2)

# ----------------------------
# Products with a constant "by hand"
# ----------------------------
n = 10
c = 2
x_vec = np.arange(1, n+1)
x_i = c * x_vec[0]
for ii in range(1, n):
    x_i *= c * x_vec[ii]
result_1 = x_i

x_vec = np.arange(1, n+1)
x_i = x_vec[0]
for ii in range(1, n):
    x_i *= x_vec[ii]
result_2 = (c**n) * x_i
print("Product with constant by hand:", result_1, "and", result_2)

# ----------------------------
# Factorials
# ----------------------------
x_val = 5
x_fac = x_val
for ii in range(1, x_val):
    x_fac *= (x_val - ii)
result_1 = x_fac
result_2 = factorial(x_val)
print("Factorial by hand:", result_1, "Built-in factorial:", result_2)

# ----------------------------
# Modulo
# ----------------------------
result = 100 % 30
print("100 mod 30 =", result)
result = 14 % 4
print("14 mod 4 =", result)

# ----------------------------
# Log and exp examples
# ----------------------------
x_val = 10
result_1 = np.log10(x_val)
result_2 = 10**(np.log10(x_val))
print("log10(10) =", result_1, "10^(log10(10)) =", result_2)
result_1 = np.exp(np.log(x_val))
print("exp(log(10)) =", result_1)

# ----------------------------
# Properties of exponents and logs
# ----------------------------
a = 5; x_val = 2; y_val = 3
print("a^x * a^y =", a**x_val * a**y_val, "and a^(x+y) =", a**(x_val+y_val))
print("a^(-x) =", a**(-x_val), "and 1/a^x =", 1/(a**x_val))
print("a^x / a^y =", a**x_val / a**y_val, "and a^(x-y) =", a**(x_val-y_val))
print("(a^x)^y =", (a**x_val)**y_val, "and a^(x*y) =", a**(x_val*y_val))
print("a^0 =", a**0)
print("log(x*y) =", np.log(x_val*y_val), "and log(x)+log(y) =", np.log(x_val)+np.log(y_val))
print("log(x^y) =", np.log(x_val**y_val), "and y*log(x) =", y_val*np.log(x_val))
print("log(1/x) =", np.log(1/x_val), "and -log(x) =", -np.log(x_val))
print("log(x/y) =", np.log(x_val/y_val), "and log(x)-log(y) =", np.log(x_val)-np.log(y_val))
print("log(1) =", np.log(1))

# ----------------------------
# Quadratic formula: Solve x^2 + 3x - 4 = 0
# ----------------------------
a_coef = 1; b_coef = 3; c_coef = -4
x_p = (-b_coef + np.sqrt(b_coef**2 - 4*a_coef*c_coef)) / (2*a_coef)
x_m = (-b_coef - np.sqrt(b_coef**2 - 4*a_coef*c_coef)) / (2*a_coef)
print("Quadratic formula solutions:", x_p, x_m)

# ----------------------------
# 3D Surface Plots: Functions
# ----------------------------

# Paraboloid of One Sheet
a_const = 2; b_const = 2; numPts = 30; z_shift = 1; scale = 1
x_0 = np.linspace(-2*a_const, 2*a_const, numPts)
y_0 = np.linspace(-2*b_const, 2*b_const, numPts)
X, Y = np.meshgrid(x_0, y_0)
Z = scale * (z_shift + X**2 / a_const**2 + Y**2 / b_const**2)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
ax.contour(X, Y, Z, zdir='z', offset=np.min(Z), cmap='viridis')
ax.set_title('Paraboloid of One Sheet')
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
plt.savefig("../figures/paraboloid_one_sheet.pdf")

# Elliptical Paraboloid
r, theta = np.meshgrid(np.arange(0, 5.05, 0.05), np.arange(0, 2*np.pi+0.01, np.pi/50))
x_val = r * np.cos(theta)
y_val = r * np.sin(theta)
z_val = x_val**2 + y_val**2
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x_val, y_val, z_val, cmap='viridis')
ax.contour(x_val, y_val, z_val, zdir='z', offset=np.min(z_val), cmap='viridis')
ax.set_title('Elliptical Paraboloid')
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
plt.savefig("../figures/elliptical_paraboloid.pdf")

# Hyperbolic Paraboloid
x_line = np.arange(-10, 10.5, 0.5)
y_line = np.arange(-10, 10.5, 0.5)
X, Y = np.meshgrid(x_line, y_line)
Z = X**2 - Y**2
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_title('Hyperbolic Paraboloid')
ax.set_xlabel('y'); ax.set_ylabel('x'); ax.set_zlabel('z')
plt.savefig("../figures/hyperbolic_paraboloid.pdf")

plt.show()
