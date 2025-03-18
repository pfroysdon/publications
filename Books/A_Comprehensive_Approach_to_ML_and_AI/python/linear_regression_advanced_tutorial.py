import numpy as np
import matplotlib.pyplot as plt

# Define data: time instants and true function (cubic)
x = np.arange(1, 101)
y = 1.07e-4 * x**3 - 0.0088 * x**2 + 0.3 * x + 2.1

# Generate noisy data
np.random.seed(1)
mu = 0
sigma = 2
y_tilde = y + (sigma * np.random.randn(len(x)) + mu)

# Plot raw data
plt.figure()
plt.plot(x, y, '.-', label='True Function')
plt.plot(x, y_tilde, 'o-', label='Noisy Data')
plt.xlabel('Time (day)')
plt.ylabel('Stock Value ($)')
plt.title('Raw Data')
plt.legend()
plt.xlim([0, 100])
plt.ylim([-10, 60])
plt.show()

# Linear (1st order) fit using least squares
H1 = np.vstack([x, np.ones_like(x)]).T
beta_ord1 = np.linalg.inv(H1.T @ H1) @ (H1.T @ y_tilde)
f1 = beta_ord1[0] * x + beta_ord1[1]
e1 = np.linalg.norm(y_tilde - f1)

# Quadratic (2nd order) fit using least squares
H2 = np.vstack([x**2, x, np.ones_like(x)]).T
beta_ord2 = np.linalg.inv(H2.T @ H2) @ (H2.T @ y_tilde)
f2 = beta_ord2[0] * x**2 + beta_ord2[1] * x + beta_ord2[2]
e2 = np.linalg.norm(y_tilde - f2)

# Cubic (3rd order) fit using least squares
H3 = np.vstack([x**3, x**2, x, np.ones_like(x)]).T
beta_ord3 = np.linalg.inv(H3.T @ H3) @ (H3.T @ y_tilde)
f3 = beta_ord3[0] * x**3 + beta_ord3[1] * x**2 + beta_ord3[2] * x + beta_ord3[3]
e3 = np.linalg.norm(y_tilde - f3)

# Cubic fit using Weighted Least Squares (WLS)
W = (2**2) * np.eye(len(x))
beta_ord3_wls = np.linalg.inv(H3.T @ W @ H3) @ (H3.T @ W @ y_tilde)
f3_wls = beta_ord3_wls[0] * x**3 + beta_ord3_wls[1] * x**2 + beta_ord3_wls[2] * x + beta_ord3_wls[3]
e3_wls = np.linalg.norm(y_tilde - f3_wls)

# Display coefficients and error norms
print("First-order fit coefficients:")
print(f"  Beta 1: {beta_ord1[0]:8.4f}")
print(f"  Beta 2: {beta_ord1[1]:8.4f}\n")

print("Second-order fit coefficients:")
print(f"  Beta 1: {beta_ord2[0]:8.4f}")
print(f"  Beta 2: {beta_ord2[1]:8.4f}")
print(f"  Beta 3: {beta_ord2[2]:8.4f}\n")

print("Third-order fit coefficients:")
print(f"  Beta 1: {beta_ord3[0]:8.4f}")
print(f"  Beta 2: {beta_ord3[1]:8.4f}")
print(f"  Beta 3: {beta_ord3[2]:8.4f}")
print(f"  Beta 4: {beta_ord3[3]:8.4f}\n")

print(f"First-order norm(error)  = {e1:8.4f}")
print(f"Second-order norm(error) = {e2:8.4f}")
print(f"Third-order norm(error)  = {e3:8.4f}")
print(f"Difference in third-order norm (LS - WLS) = {e3 - e3_wls:8.4f}")

# Plot fits for visual comparison
plt.figure()
plt.plot(x, y_tilde, 'o', label='Noisy Data', markersize=4)
plt.plot(x, f1, '--', label='1st-order fit')
plt.xlabel('Time (day)')
plt.ylabel('Stock Value ($)')
plt.title('1st-order (Linear) Fit')
plt.legend()
plt.xlim([0, 100])
plt.ylim([-10, 60])
plt.show()

plt.figure()
plt.plot(x, y_tilde, 'o', label='Noisy Data', markersize=4)
plt.plot(x, f1, '--', label='1st-order fit')
plt.plot(x, f2, ':', label='2nd-order fit')
plt.xlabel('Time (day)')
plt.ylabel('Stock Value ($)')
plt.title('1st and 2nd Order Fits')
plt.legend()
plt.xlim([0, 100])
plt.ylim([-10, 60])
plt.show()

plt.figure()
plt.plot(x, y_tilde, 'o', label='Noisy Data', markersize=4)
plt.plot(x, f1, '--', label='1st-order fit')
plt.plot(x, f2, ':', label='2nd-order fit')
plt.plot(x, f3, '-.', label='3rd-order fit')
plt.xlabel('Time (day)')
plt.ylabel('Stock Value ($)')
plt.title('1st, 2nd and 3rd Order Fits')
plt.legend()
plt.xlim([0, 100])
plt.ylim([-10, 60])
plt.show()

plt.figure()
plt.plot(x, y_tilde, 'o', label='Noisy Data', markersize=4)
plt.plot(x, f3, ':', label='LS 3rd-order fit')
plt.plot(x, f3_wls, '-.', label='WLS 3rd-order fit')
plt.xlabel('Time (day)')
plt.ylabel('Stock Value ($)')
plt.title('LS vs WLS 3rd Order Fit')
plt.legend()
plt.xlim([0, 100])
plt.ylim([-10, 60])
plt.show()
