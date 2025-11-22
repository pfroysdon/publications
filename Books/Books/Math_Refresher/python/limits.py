# limits.py
# Limits
#-------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

# Create x values from -20 to 20
x = np.arange(-20, 21, 1)

# f1(x) = sqrt(x): For negative x, we set the value to NaN (replicating real-valued sqrt)
f1 = np.where(x >= 0, np.sqrt(x), np.nan)

# f2(x) = 1/x; division by zero will produce inf
with np.errstate(divide='ignore', invalid='ignore'):
    f2 = 1 / x.astype(float)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(x, f1, '.', linewidth=1)
plt.plot(x, f1, 'b')
plt.xlabel(r'$$x$$', fontsize=12)
plt.ylabel(r'$$f(x)$$', fontsize=12)
plt.title(r'$$f(x) = \sqrt{x}$$', fontsize=12)

plt.subplot(1, 2, 2)
plt.plot(x, f2, '.', linewidth=1)
plt.plot(x, f2, 'b')
plt.xlabel(r'$$x$$', fontsize=12)
plt.ylabel(r'$$f(x)$$', fontsize=12)
plt.title(r'$$f(x) = \frac{1}{x}$$', fontsize=12)

plt.savefig("../figures/limits.pdf")
plt.show()
