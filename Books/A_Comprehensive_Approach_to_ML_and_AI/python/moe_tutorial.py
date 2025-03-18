import numpy as np
import matplotlib.pyplot as plt

# --- Data Generation ---
np.random.seed(1)
N = 200
X = np.linspace(0, 4, N).reshape(N, 1)  # N x 1
noise = 0.5 * np.random.randn(N, 1)
y = np.zeros((N, 1))
for i in range(N):
    if X[i, 0] <= 2:
        y[i, 0] = 2 * X[i, 0] + noise[i, 0]
    else:
        y[i, 0] = 8 - 2 * X[i, 0] + noise[i, 0]

# Plot synthetic data
plt.figure()
plt.scatter(X, y, color='b')
plt.xlabel("x"); plt.ylabel("y")
plt.title("Synthetic Data for Mixture of Experts Regression")
plt.grid(True)
plt.show()

# --- Mixture of Experts Model Training ---
def softmax(z):
    z = z - np.max(z)
    ex = np.exp(z)
    return ex / np.sum(ex)

def moe_train(X, y, K, learning_rate, epochs):
    n, d = X.shape
    # Initialize gating network parameters (K x d and K x 1)
    gating_W = np.random.randn(K, d) * 0.01
    gating_b = np.zeros((K, 1))
    # Initialize expert parameters (linear models for each expert: K x d and K x 1)
    expert_W = np.random.randn(K, d) * 0.01
    expert_b = np.zeros((K, 1))
    
    for epoch in range(1, epochs+1):
        idx = np.random.permutation(n)
        total_loss = 0
        for i in idx:
            x = X[i, :].reshape(d, 1)  # column vector (d x 1)
            target = y[i, 0]
            # Forward pass: gating network
            s = gating_W @ x + gating_b  # (K x 1)
            g = softmax(s)  # gating weights, (K x 1)
            # Experts: linear predictions for each expert
            f = expert_W @ x + expert_b  # (K x 1)
            # Overall prediction: weighted sum
            y_hat = float(g.T @ f)
            e = y_hat - target
            loss = 0.5 * (e**2)
            total_loss += loss
            
            # Backward pass:
            # Gradients for expert parameters:
            dW_expert = e * (g @ x.T)   # (K x d)
            db_expert = e * g           # (K x 1)
            # Gradients for gating parameters:
            d_s = e * (g * (f - y_hat))  # (K x 1)
            dW_gating = d_s @ x.T       # (K x d)
            db_gating = d_s             # (K x 1)
            
            # Parameter updates:
            expert_W -= learning_rate * dW_expert
            expert_b -= learning_rate * db_expert
            gating_W -= learning_rate * dW_gating
            gating_b -= learning_rate * db_gating
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss/n:.4f}")
    model = {
        'gating': {'W': gating_W, 'b': gating_b},
        'expert': {'W': expert_W, 'b': expert_b}
    }
    return model

def moe_predict(model, X):
    n, d = X.shape
    y_pred = np.zeros((n, 1))
    for i in range(n):
        x = X[i, :].reshape(d, 1)
        s = model['gating']['W'] @ x + model['gating']['b']
        g = softmax(s)
        f = model['expert']['W'] @ x + model['expert']['b']
        y_pred[i, 0] = float(g.T @ f)
    return y_pred

def moe_predict_gated(model, x):
    # x is a d x 1 column vector
    s = model['gating']['W'] @ x + model['gating']['b']
    gating_out = softmax(s)
    expert_out = model['expert']['W'] @ x + model['expert']['b']
    y_hat = float(gating_out.T @ expert_out)
    return y_hat, expert_out, gating_out

# Train the model
K = 2
learning_rate = 0.01
epochs = 10000
model = moe_train(X, y, K, learning_rate, epochs)

# Predict on training data
y_pred = moe_predict(model, X)
plt.figure()
plt.scatter(X, y, color='b', label='True Data')
plt.plot(X, y_pred, 'r-', linewidth=2, label='Prediction')
plt.xlabel("x"); plt.ylabel("y")
plt.title("Mixture of Experts: True Data vs. Model Prediction")
plt.legend()
plt.grid(True)
plt.show()

# Visualize Experts and Gating Network outputs
expert_preds = np.zeros((N, K))
gating_weights = np.zeros((N, K))
for i in range(N):
    x_val = X[i, :].reshape(-1, 1)
    _, expert_out, gating_out = moe_predict_gated(model, x_val)
    expert_preds[i, :] = expert_out.flatten()
    gating_weights[i, :] = gating_out.flatten()

plt.figure()
plt.plot(X, expert_preds[:, 0], 'g--', linewidth=2, label='Expert 1')
plt.plot(X, expert_preds[:, 1], 'm--', linewidth=2, label='Expert 2')
plt.xlabel("x"); plt.ylabel("Expert Prediction")
plt.title("Experts' Predictions")
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(X, gating_weights[:, 0], 'g-', linewidth=2, label='Weight Expert 1')
plt.plot(X, gating_weights[:, 1], 'm-', linewidth=2, label='Weight Expert 2')
plt.xlabel("x"); plt.ylabel("Gating Weight")
plt.title("Gating Network Outputs")
plt.legend()
plt.grid(True)
plt.show()
