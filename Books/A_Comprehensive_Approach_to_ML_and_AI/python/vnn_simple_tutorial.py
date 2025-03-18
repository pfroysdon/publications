import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
N = 100
# Class +1: centered at (2,2)
X1 = np.random.randn(N, 2) + 2
# Class -1: centered at (-2,-2)
X2 = np.random.randn(N, 2) - 2
X = np.vstack([X1, X2])
y = np.vstack([np.ones((N, 1)), -np.ones((N, 1))])

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

def vnn_train(X, y, hidden_size, learning_rate, epochs):
    n, d = X.shape
    # Initialize weights.
    W1 = np.random.randn(hidden_size, d) * 0.01
    b1 = np.zeros((hidden_size, 1))
    W2 = np.random.randn(1, hidden_size) * 0.01
    b2 = 0.0
    
    loss_history = np.zeros(epochs)
    for epoch in range(epochs):
        # Forward pass.
        Z1 = W1 @ X.T + b1  # shape: (hidden_size, n)
        A1 = relu(Z1)
        Z2 = W2 @ A1 + b2  # shape: (1, n)
        A2 = np.tanh(Z2)  # outputs in [-1,1]
        loss = 0.5 * np.mean((A2 - y.T)**2)
        loss_history[epoch] = loss
        
        # Backward pass.
        dA2 = (A2 - y.T) / n  # (1, n)
        dZ2 = dA2 * tanh_derivative(Z2)
        dW2 = dZ2 @ A1.T
        db2 = np.sum(dZ2, axis=1, keepdims=True)
        
        dA1 = W2.T @ dZ2
        dZ1 = dA1 * relu_derivative(Z1)
        dW1 = dZ1 @ X
        db1 = np.sum(dZ1, axis=1, keepdims=True)
        
        # Parameter update.
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        
        if (epoch+1) % 1000 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
    model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    return model, loss_history

def vnn_predict(model, X):
    n = X.shape[0]
    Z1 = model['W1'] @ X.T + model['b1']
    A1 = relu(Z1)
    Z2 = model['W2'] @ A1 + model['b2']
    A2 = np.tanh(Z2)
    return A2.T  # shape (n,1)

model, loss_history = vnn_train(X, y, hidden_size=10, learning_rate=0.02, epochs=500)
y_pred = vnn_predict(model, X)
accuracy = np.mean(np.sign(y_pred) == y) * 100
print(f"Training Accuracy: {accuracy:.2f}%")

# Visualize decision boundary.
x1_range = np.linspace(X[:,0].min()-1, X[:,0].max()+1, 100)
x2_range = np.linspace(X[:,1].min()-1, X[:,1].max()+1, 100)
xx, yy = np.meshgrid(x1_range, x2_range)
grid_points = np.c_[xx.ravel(), yy.ravel()]
preds = vnn_predict(model, grid_points)
Z = np.reshape(np.sign(preds), xx.shape)

plt.figure()
plt.scatter(X[y.flatten()==1, 0], X[y.flatten()==1, 1], color='b', label='Class +1')
plt.scatter(X[y.flatten()==-1, 0], X[y.flatten()==-1, 1], color='r', label='Class -1')
plt.contourf(xx, yy, Z, levels=[-np.inf, 0, np.inf], alpha=0.3, colors=['r', 'b'])
plt.title("VNN Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(loss_history, linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss History")
plt.grid(True)
plt.show()
