#!/usr/bin/env python3
"""
kdeTutorial_simple.py
---------------------
This tutorial demonstrates simple kernel density estimation.
First, it shows KDE on a small dataset (SixMPG), then it loads automotive MPG data
and fits KDEs with different bandwidths.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import gaussian_kde

# --- Part 1: Small Dataset ---
SixMPG = np.array([13,15,23,29,32,34])
plt.figure()
plt.hist(SixMPG, bins='auto', color='skyblue', edgecolor='k')
plt.title("Histogram of SixMPG")
plt.xlabel("MPG")
plt.ylabel("Count")
plt.grid(True)
plt.show()

# Fit a KDE with bandwidth 4
kde_six = gaussian_kde(SixMPG, bw_method=4/np.std(SixMPG, ddof=1))
x = np.linspace(0,45,400)
ySix = kde_six(x)
plt.figure()
plt.plot(x, ySix, 'k-', linewidth=2, label="KDE Bandwidth=4")
plt.title("KDE for SixMPG")
plt.xlabel("X"); plt.ylabel("Estimated Density")
plt.grid(True)
# Plot individual Gaussian pdfs for each point, scaled.
for mpg in SixMPG:
    y_ind = norm.pdf(x, loc=mpg, scale=4) / 6  # scale factor 1/6
    plt.plot(x, y_ind, 'b:', linewidth=2)
plt.legend()
plt.show()

# --- Part 2: Automotive MPG Data ---
# For demonstration, we simulate automotive MPG data.
# In practice, load your dataset (e.g., from carbig.mat).
np.random.seed(1)
MPG = np.random.normal(30, 5, 100)  # simulate 100 MPG values

pd1 = gaussian_kde(MPG)           # default bandwidth
pd2 = gaussian_kde(MPG, bw_method=1/np.std(MPG, ddof=1))
pd3 = gaussian_kde(MPG, bw_method=5/np.std(MPG, ddof=1))

x_vals = np.linspace(-10, 60, 200)
y1 = pd1(x_vals)
y2 = pd2(x_vals)
y3 = pd3(x_vals)

plt.figure()
plt.plot(x_vals, y1, 'r-', linewidth=2, label="Bandwidth = Default")
plt.plot(x_vals, y2, 'k:', linewidth=2, label="Bandwidth = 1")
plt.plot(x_vals, y3, 'b--', linewidth=2, label="Bandwidth = 5")
plt.legend()
plt.xlabel("X"); plt.ylabel("Estimated Density")
plt.title("KDE for Automotive MPG")
plt.grid(True)
plt.show()
