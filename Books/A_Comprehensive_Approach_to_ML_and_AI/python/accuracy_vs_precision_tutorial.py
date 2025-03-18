#!/usr/bin/env python3
"""
accuracy_vs_precision.py
-------------------------
This script reproduces the MATLAB "accuracy_vs_precision.m" tutorial.
It creates a figure with 2×4 subplots. The top row shows various circle plots with
'X' markers and dashed lines; the bottom row shows example distributions.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def circle(x0, y0, r, N=200):
    """Return x and y coordinates of a circle centered at (x0,y0) with radius r."""
    theta = np.linspace(0, 2*np.pi, N)
    xC = x0 + r * np.cos(theta)
    yC = y0 + r * np.sin(theta)
    return xC, yC

def main():
    radii = np.array([1, 2, 3, 4, 5])
    nCirclePoints = 200
    xvals = np.linspace(-10, 10, 200)

    # Create figure with 2 rows and 4 columns of subplots
    fig, axs = plt.subplots(2, 4, figsize=(12, 6))
    
    # --- Subplot (a) Top ---
    ax = axs[0, 0]
    ax.set_title('(a)')
    ax.set_aspect('equal')
    for r in radii:
        xC, yC = circle(0, 0, r, nCirclePoints)
        ax.plot(xC, yC, 'k')
    # Plot several 'X' markers at specified angles and radii
    markers = [(1.5, 0), (4.5, 80), (4.5, 150), (2.5, 200), (4.5, 300)]
    for r_val, theta_deg in markers:
        theta_rad = np.deg2rad(theta_deg)
        xX = r_val * np.cos(theta_rad)
        yX = r_val * np.sin(theta_rad)
        ax.plot(xX, yX, 'kx', markersize=10, linewidth=2)
    ax.axvline(0, color='r', linestyle='--', linewidth=1.5)
    ax.axvline(0.75, color='g', linestyle='--', linewidth=1.5)
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_xlabel('X'); ax.set_ylabel('Y')
    ax.grid(True)

    # --- Subplot (a) Bottom ---
    ax = axs[1, 0]
    mu_a = 0.75
    sigma_a = 2
    yvals_a = norm.pdf(xvals, loc=mu_a, scale=sigma_a)
    ax.plot(xvals, yvals_a, 'k', linewidth=1.5)
    ax.axvline(0, color='r', linestyle='--', linewidth=1.5)
    ax.axvline(mu_a, color='g', linestyle='--', linewidth=1.5)
    ax.set_title('(a) Distribution')
    ax.set_xlabel('x'); ax.set_ylabel('PDF')
    ax.set_xlim([-5, 5])
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)

    # --- Subplot (b) Top ---
    ax = axs[0, 1]
    ax.set_title('(b)')
    ax.set_aspect('equal')
    for r in radii:
        xC, yC = circle(0, 0, r, nCirclePoints)
        ax.plot(xC, yC, 'k')
    points = [(2, 0), (2, 90), (2, 180), (2, 270), (0, 0)]
    for r_val, theta_deg in points:
        theta_rad = np.deg2rad(theta_deg)
        xX = r_val * np.cos(theta_rad)
        yX = r_val * np.sin(theta_rad)
        ax.plot(xX, yX, 'kx', markersize=10, linewidth=2)
    ax.axvline(0, color='r', linestyle='--', linewidth=1.5)
    ax.axvline(0, color='g', linestyle='--', linewidth=1.5)
    ax.set_xlim([-5, 5]); ax.set_ylim([-5, 5])
    ax.set_xlabel('X'); ax.set_ylabel('Y')
    ax.grid(True)

    # --- Subplot (b) Bottom ---
    ax = axs[1, 1]
    mu_b = 0
    sigma_b = 1
    yvals_b = norm.pdf(xvals, loc=mu_b, scale=sigma_b)
    ax.plot(xvals, yvals_b, 'k', linewidth=1.5)
    ax.axvline(0, color='r', linestyle='--', linewidth=1.5)
    ax.axvline(mu_b, color='g', linestyle='--', linewidth=1.5)
    ax.set_title('(b) Distribution')
    ax.set_xlabel('x'); ax.set_ylabel('PDF')
    ax.set_xlim([-5, 5])
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)

    # --- Subplot (c) Top ---
    ax = axs[0, 2]
    ax.set_title('(c)')
    ax.set_aspect('equal')
    for r in radii:
        xC, yC = circle(0, 0, r, nCirclePoints)
        ax.plot(xC, yC, 'k')
    markers_c = [(4.5, 100), (4.5, 102), (4.5, 104), (4.5, 106), (4.5, 108)]
    for r_val, theta_deg in markers_c:
        theta_rad = np.deg2rad(theta_deg)
        xX = r_val * np.cos(theta_rad)
        yX = r_val * np.sin(theta_rad)
        ax.plot(xX, yX, 'kx', markersize=10, linewidth=2)
    ax.axvline(0, color='r', linestyle='--', linewidth=1.5)
    ax.axvline(-1.2, color='g', linestyle='--', linewidth=1.5)
    ax.set_xlim([-5, 5]); ax.set_ylim([-5, 5])
    ax.set_xlabel('X'); ax.set_ylabel('Y')
    ax.grid(True)

    # --- Subplot (c) Bottom ---
    ax = axs[1, 2]
    mu_c = -1.2
    sigma_c = 0.2
    yvals_c = norm.pdf(xvals, loc=mu_c, scale=sigma_c)
    ax.plot(xvals, yvals_c, 'k', linewidth=1.5)
    ax.axvline(0, color='r', linestyle='--', linewidth=1.5)
    ax.axvline(mu_c, color='g', linestyle='--', linewidth=1.5)
    ax.set_title('(c) Distribution')
    ax.set_xlabel('x'); ax.set_ylabel('PDF')
    ax.set_xlim([-5, 5])
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)

    # --- Subplot (d) Top ---
    ax = axs[0, 3]
    ax.set_title('(d)')
    ax.set_aspect('equal')
    for r in radii:
        xC, yC = circle(0, 0, r, nCirclePoints)
        ax.plot(xC, yC, 'k')
    points_d = [(0.5, 0), (0.5, 90), (0.5, 180), (0.5, 270), (0, 0)]
    for r_val, theta_deg in points_d:
        theta_rad = np.deg2rad(theta_deg)
        xX = r_val * np.cos(theta_rad)
        yX = r_val * np.sin(theta_rad)
        ax.plot(xX, yX, 'kx', markersize=10, linewidth=2)
    ax.axvline(0, color='r', linestyle='--', linewidth=1.5)
    ax.axvline(0, color='g', linestyle='--', linewidth=1.5)
    ax.set_xlim([-5, 5]); ax.set_ylim([-5, 5])
    ax.set_xlabel('X'); ax.set_ylabel('Y')
    ax.grid(True)

    # --- Subplot (d) Bottom ---
    ax = axs[1, 3]
    mu_d = 0
    sigma_d = 0.2
    yvals_d = norm.pdf(xvals, loc=mu_d, scale=sigma_d)
    ax.plot(xvals, yvals_d, 'k', linewidth=1.5)
    ax.axvline(0, color='r', linestyle='--', linewidth=1.5)
    ax.axvline(0, color='g', linestyle='--', linewidth=1.5)
    ax.set_title('(d) Distribution')
    ax.set_xlabel('x'); ax.set_ylabel('PDF')
    ax.set_xlim([-5, 5])
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
