import numpy as np
import matplotlib.pyplot as plt

def mlp_train(X, Y, num_hidden, alpha, epochs):
    # X: (N, M) features; Y: (N, 1) labels
    N, M = X.shape
    # He initialization for weights
    W1 = np.random.randn(M, num_hidden) * np.sqrt(2/M)
    b1 = np.zeros((1, num_hidden))
    W2 = np.random.randn(num_hidden, 1) * np.sqrt(2/num_hidden)
    b2 = 0
    
    for epoch in range(epochs):
        # Forward propagation
        Z1 = X @ W1 + b1  # (N, num_hidden)
        A1 = np.maximum(0, Z1)  # ReLU activation
        Z2 = A1 @ W2 + b2      # (N, 1)
        A2 = 1 / (1 + np.exp(-Z2))  # Sigmoid activation
        
        # Compute loss (MSE)
        loss = 0.5 * np.mean((A2 - Y)**2)
        
        # Backpropagation
        dZ2 = A2 - Y  # (N,1)
        dW2 = (A1.T @ dZ2) / N
        db2 = np.mean(dZ2)
        dA1 = dZ2 @ W2.T  # (N, num_hidden)
        dZ1 = dA1 * (A1 > 0)  # ReLU derivative
        dW1 = (X.T @ dZ1) / N
        db1 = np.mean(dZ1, axis=0, keepdims=True)
        
        # Update parameters
        W1 -= alpha * dW1
        b1 -= alpha * db1
        W2 -= alpha * dW2
        b2 -= alpha * db2
        
    model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    return model

def mlp_predict(model, X):
    Z1 = X @ model['W1'] + model['b1']
    A1 = np.maximum(0, Z1)
    Z2 = A1 @ model['W2'] + model['b2']
    A2 = 1 / (1 + np.exp(-Z2))
    # Convert probabilities to binary labels using 0.5 threshold
    return (A2 >= 0.5).astype(int)

if __name__ == '__main__':
    np.random.seed(42)
    num_samples = 200
    # Class 0: centered at 1, Class 1: centered at -1
    X1 = np.random.randn(num_samples//2, 2) + 1
    Y1 = np.zeros((num_samples//2, 1))
    X2 = np.random.randn(num_samples//2, 2) - 1
    Y2 = np.ones((num_samples//2, 1))
    
    X = np.vstack([X1, X2])
    Y = np.vstack([Y1, Y2])
    
    # Shuffle dataset
    indices = np.random.permutation(num_samples)
    X = X[indices]
    Y = Y[indices]
    
    # Plot dataset
    plt.figure()
    plt.scatter(X[Y.flatten()==0, 0], X[Y.flatten()==0, 1], c='b', label='Class 0')
    plt.scatter(X[Y.flatten()==1, 0], X[Y.flatten()==1, 1], c='r', label='Class 1')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Synthetic Binary Classification Dataset")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Normalize features
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    
    num_hidden = 5
    alpha = 0.1
    epochs = 1000
    model = mlp_train(X, Y, num_hidden, alpha, epochs)
    print("Training complete!")
    
    Y_pred = mlp_predict(model, X)
    accuracy = np.mean(Y_pred == Y) * 100
    print(f"Model Accuracy: {accuracy:.2f}%")
    
    # Create mesh grid for decision boundary visualization
    x1_min, x1_max = X[:, 0].min()-1, X[:, 0].max()+1
    x2_min, x2_max = X[:, 1].min()-1, X[:, 1].max()+1
    xx, yy = np.meshgrid(np.linspace(x1_min, x1_max, 100),
                         np.linspace(x2_min, x2_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_preds = mlp_predict(model, grid).reshape(xx.shape)
    
    plt.figure()
    plt.contourf(xx, yy, grid_preds, alpha=0.3, levels=[-0.5, 0.5, 1.5], cmap=plt.cm.Paired)
    plt.scatter(X[Y.flatten()==0, 0], X[Y.flatten()==0, 1], c='r', label='Class 0')
    plt.scatter(X[Y.flatten()==1, 0], X[Y.flatten()==1, 1], c='b', label='Class 1')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("MLP Decision Boundary")
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()
