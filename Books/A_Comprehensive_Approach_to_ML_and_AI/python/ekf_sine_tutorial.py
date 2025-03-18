#!/usr/bin/env python3
"""
ekfTutorial_sine.py
-------------------
Extended Kalman Filter (EKF) for 2D tracking of a sinusoidal trajectory.
The target moves with x(t) = v_x * t and y(t) = A * sin(omega * t).
Noisy measurements (range and bearing) are generated and the EKF estimates the state.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

np.random.seed(1)

# Simulation parameters
dt = 0.1
T = 20
t = np.arange(0, T+dt, dt)
N = len(t)

# True state simulation
v_x = 1.0
A = 2.0
omega = 0.5
x_true = np.zeros((4, N))
for k in range(N):
    x_true[0, k] = v_x * t[k]
    x_true[1, k] = A * np.sin(omega * t[k])
    x_true[2, k] = v_x
    x_true[3, k] = A * omega * np.cos(omega * t[k])

# Measurement simulation (range and bearing)
R_meas = np.diag([0.1**2, (np.deg2rad(5))**2])
z = np.zeros((2, N))
for k in range(N):
    pos = x_true[0:2, k]
    range_true = np.linalg.norm(pos)
    bearing_true = np.arctan2(pos[1], pos[0])
    noise = np.random.multivariate_normal([0,0], R_meas)
    z[:, k] = np.array([range_true, bearing_true]) + noise

# EKF initialization
F = np.array([[1, 0, dt, 0],
              [0, 1, 0, dt],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])
P = 10 * np.eye(4)
Q = np.diag([0.005, 0.005, 0.05, 0.05])
x_est = np.zeros((4, N))
x_est[:, 0] = np.array([0.2, -0.2, 0.8, 0.6])

def h(x):
    x_pos, y_pos = x[0], x[1]
    return np.array([np.sqrt(x_pos**2 + y_pos**2),
                     np.arctan2(y_pos, x_pos)])

def H_jacobian(x):
    x_pos, y_pos = x[0], x[1]
    range_val = np.sqrt(x_pos**2 + y_pos**2)
    if range_val < 1e-4:
        return np.zeros((2,4))
    d_range_dx = x_pos / range_val
    d_range_dy = y_pos / range_val
    d_bearing_dx = -y_pos / (range_val**2)
    d_bearing_dy = x_pos / (range_val**2)
    H_jac = np.array([[d_range_dx, d_range_dy, 0, 0],
                      [d_bearing_dx, d_bearing_dy, 0, 0]])
    return H_jac

for k in range(N-1):
    # Prediction
    x_pred = F @ x_est[:, k]
    P_pred = F @ P @ F.T + Q
    # Measurement prediction
    z_pred = h(x_pred)
    H_j = H_jacobian(x_pred)
    y_innov = z[:, k+1] - z_pred
    # Normalize bearing innovation to [-pi, pi]
    y_innov[1] = (y_innov[1] + np.pi) % (2*np.pi) - np.pi
    S = H_j @ P_pred @ H_j.T + R_meas
    K = P_pred @ H_j.T @ np.linalg.inv(S)
    x_est[:, k+1] = x_pred + K @ y_innov
    P = (np.eye(4) - K @ H_j) @ P_pred

# Plotting
plt.figure()
plt.subplot(2,1,1)
plt.plot(t, x_true[0,:], 'b-', label="True x")
plt.plot(t, x_est[0,:], 'r--', label="Estimated x")
plt.xlabel("Time (s)"); plt.ylabel("x position (m)")
plt.title("Cart Position Tracking")
plt.legend(); plt.grid(True)

plt.subplot(2,1,2)
plt.plot(t, x_true[1,:], 'b-', label="True y")
plt.plot(t, x_est[1,:], 'r--', label="Estimated y")
plt.xlabel("Time (s)"); plt.ylabel("y position (m)")
plt.title("Pole Position Tracking")
plt.legend(); plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure()
plt.subplot(2,1,1)
plt.plot(t, x_true[2,:], 'b-', label="True vx")
plt.plot(t, x_est[2,:], 'r--', label="Estimated vx")
plt.xlabel("Time (s)"); plt.ylabel("x velocity (m/s)")
plt.title("Cart Velocity Tracking")
plt.legend(); plt.grid(True)

plt.subplot(2,1,2)
plt.plot(t, x_true[3,:], 'b-', label="True vy")
plt.plot(t, x_est[3,:], 'r--', label="Estimated vy")
plt.xlabel("Time (s)"); plt.ylabel("y velocity (m/s)")
plt.title("Pole Velocity Tracking")
plt.legend(); plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(x_true[0,:], x_true[1,:], 'b-', label="True Trajectory")
plt.plot(x_est[0,:], x_est[1,:], 'r--', label="Estimated Trajectory")
plt.xlabel("x position (m)"); plt.ylabel("y position (m)")
plt.title("2D Trajectory Tracking")
plt.legend(); plt.grid(True)
plt.show()
