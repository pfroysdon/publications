import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(1)

# Parameters
d = 10           # Input dimension
N = 200          # Number of samples
r = 2            # Rank of adaptation (low rank update)
learningRate = 0.01
numEpochs = 1000

# Generate Data
# X: each column is a sample (d x N)
X = np.random.randn(d, N)
# Define the ideal (target) weight vector W_target (1 x d)
W_target = np.linspace(1, 2, d)  # e.g., vector from 1 to 2
# Generate outputs with some noise
Y = W_target @ X + 0.1 * np.random.randn(1, N)

# Pre-trained Weight (W0)
W0 = W_target - 0.5  # W0 is 0.5 less than W_target

# LoRA Initialization: low-rank update parameters B (1 x r) and A (r x d)
B = np.random.randn(1, r) * 0.01
A = np.random.randn(r, d) * 0.01

# Training Loop (Adaptation using LoRA)
lossHistory = np.zeros(numEpochs)
for epoch in range(numEpochs):
    # Forward pass: Compute adapted weight
    W = W0 + B @ A  # (1 x d)
    # Compute predictions
    Yhat = W @ X  # (1 x N)
    # Compute mean squared error loss
    loss = np.mean((Y - Yhat)**2)
    lossHistory[epoch] = loss
    
    # Compute gradient of loss with respect to W
    error = Yhat - Y  # (1 x N)
    gradW = (error @ X.T) / N  # (1 x d)
    
    # Using chain rule:
    gradB = gradW @ A.T   # (1 x r)
    gradA = B.T @ gradW   # (r x d)
    
    # Update parameters using gradient descent
    B = B - learningRate * gradB
    A = A - learningRate * gradA
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

# Plot Training Loss
plt.figure()
plt.plot(np.arange(1, numEpochs+1), lossHistory, linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error Loss')
plt.title('LoRA Adaptation Training Loss')
plt.grid(True)
plt.show()

# Compare Adapted Weight to Target
W_adapted = W0 + B @ A
print("Pre-trained weight W0:\n", W0)
print("Target weight W_target:\n", W_target)
print("Adapted weight W0+B*A:\n", W_adapted)

# Predict on a Test Sample
x_test = np.random.randn(d, 1)
y_pred = W_adapted @ x_test
y_true = W_target @ x_test
print(f"Test sample prediction: {y_pred[0]:.4f} (target: {y_true[0]:.4f})")
