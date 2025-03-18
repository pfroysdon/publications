import numpy as np
import matplotlib.pyplot as plt

def least_squares_filter(H, Y):
    """Compute the least squares estimate using the normal equation."""
    # X_est = (H^T * H)^{-1} * H^T * Y
    return np.linalg.inv(H.T @ H) @ (H.T @ Y)

# Generate synthetic data for position estimation
N = 10  # Number of measurements
true_position = 5  # True state

# Observation matrix: H is an N x 1 column vector of ones (direct observation)
H = np.ones((N, 1))

# Generate noisy measurements (add Gaussian noise)
np.random.seed(42)
Y = true_position + np.random.randn(N, 1) * 0.5

# Apply Least Squares Filter
X_est = least_squares_filter(H, Y)

# Display results
print(f"True Position: {true_position:.2f}")
print(f"Estimated Position: {X_est[0]:.2f}")

# Plot results
plt.figure()
plt.scatter(np.arange(1, N+1), Y, color='red', label='Noisy Measurements')
plt.plot(np.arange(1, N+1), np.ones(N) * X_est[0], color='blue', linewidth=2, label='LSF Estimate')
plt.plot(np.arange(1, N+1), np.ones(N) * true_position, 'k--', linewidth=2, label='True Position')
plt.xlabel('Measurement Index')
plt.ylabel('Position Estimate')
plt.title('Least Squares Filter Estimation')
plt.legend()
plt.grid(True)
plt.show()
