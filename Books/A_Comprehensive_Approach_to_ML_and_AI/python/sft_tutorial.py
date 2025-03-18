import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

def relu(x):
    return np.maximum(x, 0)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    x = x - np.max(x, axis=0, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)

# 1. Generate pre-training dataset (large dataset)
N_pre = 500
# Class 1: centered at (1,1)
X1_pre = np.random.randn(2, N_pre//2)*0.5 + np.array([[1], [1]])
# Class 2: centered at (3,3)
X2_pre = np.random.randn(2, N_pre//2)*0.5 + np.array([[3], [3]])
X_pre = np.hstack([X1_pre, X2_pre])  # shape: (2, N_pre)
labels_pre = np.hstack([np.ones(N_pre//2), 2*np.ones(N_pre//2)])
Y_pre = np.zeros((2, N_pre))
for i in range(N_pre):
    Y_pre[int(labels_pre[i])-1, i] = 1

# 2. Pre-train the model
input_dim = 2
hidden_dim = 10
output_dim = 2

W1 = np.random.randn(hidden_dim, input_dim) * 0.01
b1 = np.zeros((hidden_dim, 1))
W2 = np.random.randn(output_dim, hidden_dim) * 0.01
b2 = np.zeros((output_dim, 1))

lr_pre = 0.01
num_iter_pre = 3000

for iter in range(1, num_iter_pre+1):
    Z1 = W1 @ X_pre + b1  # (10, N_pre)
    H = relu(Z1)
    logits = W2 @ H + b2   # (2, N_pre)
    probs = softmax(logits)
    loss_pre = -np.mean(np.sum(Y_pre * np.log(probs+1e-8), axis=0))
    
    d_logits = probs - Y_pre
    grad_W2 = d_logits @ H.T / N_pre
    grad_b2 = np.mean(d_logits, axis=1, keepdims=True)
    
    d_H = W2.T @ d_logits
    d_Z1 = d_H * relu_derivative(Z1)
    grad_W1 = d_Z1 @ X_pre.T / N_pre
    grad_b1 = np.mean(d_Z1, axis=1, keepdims=True)
    
    W2 -= lr_pre * grad_W2
    b2 -= lr_pre * grad_b2
    W1 -= lr_pre * grad_W1
    b1 -= lr_pre * grad_b1
    
    if iter % 500 == 0:
        print(f"Pre-training Iteration {iter}, Loss: {loss_pre:.4f}")

pretrained = {'W1': W1.copy(), 'b1': b1.copy(), 'W2': W2.copy(), 'b2': b2.copy()}

# 3. Generate fine-tuning dataset (smaller downstream task)
N_ft = 100
X1_ft = np.random.randn(2, N_ft//2)*0.5 + np.array([[1.5], [1.5]])
X2_ft = np.random.randn(2, N_ft//2)*0.5 + np.array([[2.5], [2.5]])
X_ft = np.hstack([X1_ft, X2_ft])
labels_ft = np.hstack([np.ones(N_ft//2), 2*np.ones(N_ft//2)])
Y_ft = np.zeros((2, N_ft))
for i in range(N_ft):
    Y_ft[int(labels_ft[i])-1, i] = 1

# 4. Fine-tuning using the pre-trained model as initialization.
lr_ft = 0.001
num_iter_ft = 2000

for iter in range(1, num_iter_ft+1):
    Z1 = W1 @ X_ft + b1
    H = relu(Z1)
    logits = W2 @ H + b2
    probs = softmax(logits)
    loss_ft = -np.mean(np.sum(Y_ft * np.log(probs+1e-8), axis=0))
    
    d_logits = probs - Y_ft
    grad_W2 = d_logits @ H.T / N_ft
    grad_b2 = np.mean(d_logits, axis=1, keepdims=True)
    
    d_H = W2.T @ d_logits
    d_Z1 = d_H * relu_derivative(Z1)
    grad_W1 = d_Z1 @ X_ft.T / N_ft
    grad_b1 = np.mean(d_Z1, axis=1, keepdims=True)
    
    W2 -= lr_ft * grad_W2
    b2 -= lr_ft * grad_b2
    W1 -= lr_ft * grad_W1
    b1 -= lr_ft * grad_b1
    
    if iter % 500 == 0:
        print(f"Fine-tuning Iteration {iter}, Loss: {loss_ft:.4f}")

# 5. Visualize decision boundaries before and after fine-tuning.
def predict(X_input, W1, b1, W2, b2):
    Z1 = W1 @ X_input + b1
    H = relu(Z1)
    logits = W2 @ H + b2
    return softmax(logits)

# Create grid
x_min, x_max = X_ft[0].min()-1, X_ft[0].max()+1
y_min, y_max = X_ft[1].min()-1, X_ft[1].max()+1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
grid_points = np.vstack([xx.ravel(), yy.ravel()])

# Pre-trained predictions.
probs_pre = predict(grid_points, pretrained['W1'], pretrained['b1'], pretrained['W2'], pretrained['b2'])
pred_pre = np.argmax(probs_pre, axis=0) + 1
pred_pre_grid = pred_pre.reshape(xx.shape)

# Fine-tuned predictions.
probs_ft = predict(grid_points, W1, b1, W2, b2)
pred_ft = np.argmax(probs_ft, axis=0) + 1
pred_ft_grid = pred_ft.reshape(xx.shape)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.contourf(xx, yy, pred_pre_grid, alpha=0.5, cmap='jet')
plt.scatter(X_ft[0, labels_ft==1], X_ft[1, labels_ft==1], color='w', edgecolor='k', label='Class 1')
plt.scatter(X_ft[0, labels_ft==2], X_ft[1, labels_ft==2], color='k', label='Class 2')
plt.title("Pre-trained Model Decision Boundary")
plt.xlabel("Feature 1"); plt.ylabel("Feature 2")
plt.legend()

plt.subplot(1,2,2)
plt.contourf(xx, yy, pred_ft_grid, alpha=0.5, cmap='jet')
plt.scatter(X_ft[0, labels_ft==1], X_ft[1, labels_ft==1], color='w', edgecolor='k', label='Class 1')
plt.scatter(X_ft[0, labels_ft==2], X_ft[1, labels_ft==2], color='k', label='Class 2')
plt.title("Fine-tuned Model Decision Boundary")
plt.xlabel("Feature 1"); plt.ylabel("Feature 2")
plt.legend()
plt.tight_layout()
plt.show()
