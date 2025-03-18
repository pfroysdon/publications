import numpy as np
import matplotlib.pyplot as plt

def linear_regression_train_fast(X, Y):
    """
    Optimized solution using the Normal Equation.
    
    Args:
        X (np.ndarray): Feature matrix (Nx1).
        Y (np.ndarray): Target values (Nx1).
    
    Returns:
        dict: Dictionary containing the trained weights under key 'w'.
    """
    N = X.shape[0]
    # Add bias term
    X_bias = np.hstack((np.ones((N, 1)), X))
    # Compute weights using the normal equation with pseudoinverse
    w = np.linalg.pinv(X_bias.T @ X_bias) @ (X_bias.T @ Y)
    return {'w': w}

def linear_regression_train(X, Y, alpha, epochs):
    """
    Train linear regression model using gradient descent.
    
    Args:
        X (np.ndarray): Feature matrix (Nx1).
        Y (np.ndarray): Target values (Nx1).
        alpha (float): Learning rate.
        epochs (int): Number of training iterations.
    
    Returns:
        dict: Dictionary containing the trained weights under key 'w'.
    """
    N = X.shape[0]
    # Add bias term
    X_bias = np.hstack((np.ones((N, 1)), X))
    # Initialize weights (2 x 1 vector)
    w = np.zeros((2, 1))
    
    for epoch in range(epochs):
        # Compute predictions
        Y_pred = X_bias @ w
        # Compute gradient
        gradient = (X_bias.T @ (Y_pred - Y)) / N
        # Update weights
        w = w - alpha * gradient
        # Compute loss (Mean Squared Error)
        loss = np.mean((Y - Y_pred) ** 2)
        if (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch+1}, Loss: {loss:.4f}')
            
    return {'w': w}

if __name__ == '__main__':
    # Generate synthetic dataset
    np.random.seed(42)
    X = np.linspace(1, 10, 100).reshape(-1, 1)  # Feature (house size)
    Y = 3 * X + 5 + np.random.randn(100, 1) * 2   # Target (house price)
    
    # Train Linear Regression
    alpha = 0.01
    epochs = 1000
    # Uncomment the following line to train using gradient descent:
    # model = linear_regression_train(X, Y, alpha, epochs)
    model = linear_regression_train_fast(X, Y)
    
    # Predict function: add bias term to X
    X_test = np.hstack((np.ones((X.shape[0], 1)), X))
    Y_pred = X_test @ model['w']
    
    # Compute error (Mean Squared Error)
    mse = np.mean((Y - Y_pred) ** 2)
    print(f'Mean Squared Error: {mse:.4f}')
    
    # Plot dataset and regression line
    plt.figure()
    plt.scatter(X, Y, color='blue', label='Data Points')
    plt.plot(X, Y_pred, 'r-', linewidth=2, label='Best-Fit Line')
    plt.xlabel('Feature (X)')
    plt.ylabel('Target (Y)')
    plt.title('Linear Regression Fit')
    plt.legend()
    plt.grid(True)
    plt.show()
