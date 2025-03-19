# normal_distribution.py
# Normal Distribution
#-------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

plt.close('all')

# Set seed for reproducibility (similar to rng default in MATLAB)
np.random.seed(0)

# Generate two sets of random numbers from normal distributions:
# r1 ~ N(0,1) and r2 ~ N(0,2)
r1 = np.random.normal(0, 1, 1000)
r2 = np.random.normal(0, 2, 1000)

# Generate fitted curves using the normal pdf.
# Here we use a common x-range for both curves.
x_range = np.linspace(-6, 6, 200)
y1 = norm.pdf(x_range, loc=0, scale=1)
y2 = norm.pdf(x_range, loc=0, scale=2)

plt.figure()
plt.plot(x_range, y1, ':', linewidth=2, label=r'$$\mu=0,\sigma^2=1$$')
plt.plot(x_range, y2, '-.', linewidth=2, label=r'$$\mu=0,\sigma^2=2$$')
plt.xlabel(r'$$x$$', fontsize=12)
plt.ylabel(r'$$f(x)$$', fontsize=12)
plt.title(r'$$f(x)$$', fontsize=12)
plt.legend(loc='upper right', fontsize=10)
plt.xlim([-6, 6])
plt.savefig("../figures/normal_distribution.pdf")
plt.show()
