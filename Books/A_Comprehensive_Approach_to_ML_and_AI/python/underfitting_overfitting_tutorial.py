import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
N = 30
x = np.linspace(-1, 1, N).reshape(-1, 1)
noise = 0.2 * np.random.randn(N, 1)
y = np.sin(2*np.pi*x) + noise

degrees = [1, 5, 15]
x_fine = np.linspace(-1, 1, 200).reshape(-1, 1)

def polynomial_design_matrix(x, degree):
    # Returns design matrix with columns [1, x, x^2, ..., x^degree]
    N = x.shape[0]
    X_poly = np.ones((N, degree+1))
    for p in range(1, degree+1):
        X_poly[:, p] = x[:, 0]**p
    return X_poly

plt.figure(figsize=(12,4))
for i, d in enumerate(degrees):
    X_design = polynomial_design_matrix(x, d)
    # Solve for coefficients using the normal equation
    w = np.linalg.inv(X_design.T @ X_design) @ (X_design.T @ y)
    
    X_fine_design = polynomial_design_matrix(x_fine, d)
    y_pred = X_fine_design @ w
    
    y_train_pred = X_design @ w
    mse_train = np.mean((y - y_train_pred)**2)
    
    plt.subplot(1, 3, i+1)
    plt.scatter(x, y, color='b', label='Data')
    plt.plot(x_fine, y_pred, 'r-', linewidth=2, label=f'Degree {d}')
    plt.plot(x_fine, np.sin(2*np.pi*x_fine), 'k--', linewidth=1.5, label='True Function')
    plt.title(f"Degree {d} (MSE = {mse_train:.3f})")
    plt.xlabel("x"); plt.ylabel("y")
    plt.legend(loc='best')
    plt.grid(True)
plt.tight_layout()
plt.show()
