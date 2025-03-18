#!/usr/bin/env python3
"""
filteringTutorial.py
----------------------
This tutorial demonstrates low-pass, high-pass, and band-pass filtering using the bilinear transform.
A toy signal with low and high frequency components is filtered.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import bilinear, lfilter, freqz
from scipy.fft import fft, fftfreq

def cascade_filters(b1, a1, b2, a2):
    b_cascade = np.convolve(b1, b2)
    a_cascade = np.convolve(a1, a2)
    return b_cascade, a_cascade

# Parameters
T = 0.01                     # Sampling period
fs = 1 / T                   # Sampling frequency
tau = 0.05                   # Time constant for LPF
n = np.arange(0, 1001)

# Toy signal: sum of sinusoids
x = np.sin(2 * np.pi * 5 * n * T) + 0.5 * np.sin(2 * np.pi * 50 * n * T) + 50 * np.sin(2 * np.pi * 500 * n * T)

# Design LPF: H(s)=1/(tau*s+1)
b_lp, a_lp = bilinear([1], [tau, 1], fs)
# Design HPF: H(s)=(tau*s)/(tau*s+1)
b_hp, a_hp = bilinear([tau, 0], [tau, 1], fs)
# Design BPF by cascading HPF and LPF
b_bp, a_bp = cascade_filters(b_hp, a_hp, b_lp, a_lp)

# Filter the signal
y_lp = lfilter(b_lp, a_lp, x)
y_hp = lfilter(b_hp, a_hp, x)
y_bp = lfilter(b_bp, a_bp, x)

# Plot time-domain signals
plt.figure(figsize=(10,8))
plt.subplot(4,1,1)
plt.plot(n * T, x, 'k-', linewidth=1.5)
plt.title('Original Signal')
plt.xlabel('Time (s)'); plt.ylabel('Amplitude'); plt.grid(True)

plt.subplot(4,1,2)
plt.plot(n * T, y_lp, 'b-', linewidth=1.5)
plt.title('Low-Pass Filtered Signal')
plt.xlabel('Time (s)'); plt.ylabel('Amplitude'); plt.grid(True)

plt.subplot(4,1,3)
plt.plot(n * T, y_hp, 'r-', linewidth=1.5)
plt.title('High-Pass Filtered Signal')
plt.xlabel('Time (s)'); plt.ylabel('Amplitude'); plt.grid(True)

plt.subplot(4,1,4)
plt.plot(n * T, y_bp, 'm-', linewidth=1.5)
plt.title('Band-Pass Filtered Signal')
plt.xlabel('Time (s)'); plt.ylabel('Amplitude'); plt.grid(True)
plt.tight_layout()
plt.show()

# Optionally, compute and plot FFTs here...
