import numpy as np
import matplotlib.pyplot as plt

# --- Utility functions ---
def softmax(x):
    # Compute softmax for each column (samples are columns)
    x = x - np.max(x, axis=0, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)

def relu(x):
    return np.maximum(x, 0)

def relu_derivative(x):
    return (x > 0).astype(float)

# --- Data Generation ---
np.random.seed(1)
N = 200  # total number of samples
# Create two classes: Class 1 centered at (1,1) and Class 2 at (3,3)
X1 = np.random.randn(2, N//2) * 0.5 + np.array([[1], [1]])
X2 = np.random.randn(2, N//2) * 0.5 + np.array([[3], [3]])
X = np.hstack([X1, X2])  # 2 x N
labels = np.hstack([np.ones(N//2), 2*np.ones(N//2)])  # 1 or 2

# One-hot encode labels into a 2 x N matrix
Y = np.zeros((2, N))
for i in range(N):
    Y[int(labels[i]-1), i] = 1

# --- 2. Train Teacher Network ---
# Teacher architecture: 2 -> 20 -> 2
input_dim = 2
hidden_dim_teacher = 20
output_dim = 2

# Initialize teacher parameters
W1_teacher = np.random.randn(hidden_dim_teacher, input_dim) * 0.01
b1_teacher = np.zeros((hidden_dim_teacher, 1))
W2_teacher = np.random.randn(output_dim, hidden_dim_teacher) * 0.01
b2_teacher = np.zeros((output_dim, 1))

teacher_lr = 0.01
num_iter_teacher = 5000

for iter in range(1, num_iter_teacher+1):
    # Forward pass
    Z1 = W1_teacher @ X + b1_teacher  # (20 x N)
    H1 = relu(Z1)
    logits_teacher = W2_teacher @ H1 + b2_teacher  # (2 x N)
    probs_teacher = softmax(logits_teacher)
    
    # Cross-entropy loss
    loss_teacher = -np.mean(np.sum(Y * np.log(probs_teacher + 1e-8), axis=0))
    
    # Backpropagation
    d_logits = probs_teacher - Y  # (2 x N)
    grad_W2_teacher = (d_logits @ H1.T) / N
    grad_b2_teacher = np.mean(d_logits, axis=1, keepdims=True)
    
    d_H1 = W2_teacher.T @ d_logits  # (20 x N)
    d_Z1 = d_H1 * relu_derivative(Z1)  # (20 x N)
    grad_W1_teacher = (d_Z1 @ X.T) / N
    grad_b1_teacher = np.mean(d_Z1, axis=1, keepdims=True)
    
    # Update teacher parameters
    W2_teacher -= teacher_lr * grad_W2_teacher
    b2_teacher -= teacher_lr * grad_b2_teacher
    W1_teacher -= teacher_lr * grad_W1_teacher
    b1_teacher -= teacher_lr * grad_b1_teacher
    
    if iter % 500 == 0:
        print(f"Teacher Iteration {iter}, Loss: {loss_teacher:.4f}")

# Freeze teacher parameters
Teacher = {
    'W1': W1_teacher, 'b1': b1_teacher,
    'W2': W2_teacher, 'b2': b2_teacher,
    'T': 2  # temperature for distillation
}

# --- 3. Train Student Network with Distillation ---
# Student architecture: 2 -> 5 -> 2
hidden_dim_student = 5

W1_student = np.random.randn(hidden_dim_student, input_dim) * 0.01
b1_student = np.zeros((hidden_dim_student, 1))
W2_student = np.random.randn(output_dim, hidden_dim_student) * 0.01
b2_student = np.zeros((output_dim, 1))

student_lr = 0.01
num_iter_student = 5000
lmbda = 0.5  # weight for hard loss
T = Teacher['T']

for iter in range(1, num_iter_student+1):
    # Teacher forward pass with temperature
    Z1_teacher = Teacher['W1'] @ X + Teacher['b1']
    H1_teacher = relu(Z1_teacher)
    logits_teacher = Teacher['W2'] @ H1_teacher + Teacher['b2']
    logits_teacher_T = logits_teacher / T
    soft_targets = softmax(logits_teacher_T)
    
    # Student forward pass with temperature
    Z1_student = W1_student @ X + b1_student
    H1_student = relu(Z1_student)
    logits_student = W2_student @ H1_student + b2_student
    logits_student_T = logits_student / T
    student_soft = softmax(logits_student_T)
    
    # Student hard predictions (without temperature)
    student_probs = softmax(logits_student)
    
    # Hard loss: cross-entropy loss
    hard_loss = -np.mean(np.sum(Y * np.log(student_probs + 1e-8), axis=0))
    
    # Distillation loss: KL divergence (per sample)
    kl_div = np.sum(soft_targets * (np.log(soft_targets + 1e-8) - np.log(student_soft + 1e-8)), axis=0)
    distill_loss = np.mean(kl_div)
    
    loss_student = lmbda * hard_loss + (1 - lmbda) * (T**2) * distill_loss
    
    # Backpropagation for student
    d_logits_hard = student_probs - Y  # (2 x N)
    d_logits_distill = student_soft - soft_targets  # (2 x N)
    d_logits_distill = (1 / T) * d_logits_distill
    d_logits_student = lmbda * d_logits_hard + (1 - lmbda) * (T**2) * d_logits_distill
    
    grad_W2_student = (d_logits_student @ H1_student.T) / N
    grad_b2_student = np.mean(d_logits_student, axis=1, keepdims=True)
    
    d_H1_student = W2_student.T @ d_logits_student
    d_Z1_student = d_H1_student * relu_derivative(Z1_student)
    grad_W1_student = (d_Z1_student @ X.T) / N
    grad_b1_student = np.mean(d_Z1_student, axis=1, keepdims=True)
    
    # Update student parameters
    W2_student -= student_lr * grad_W2_student
    b2_student -= student_lr * grad_b2_student
    W1_student -= student_lr * grad_W1_student
    b1_student -= student_lr * grad_b1_student
    
    if iter % 500 == 0:
        print(f"Student Iteration {iter}, Hard Loss: {hard_loss:.4f}, Distill Loss: {distill_loss:.4f}, Combined Loss: {loss_student:.4f}")

# --- 4. Evaluate and Visualize Decision Boundaries ---
# Create a grid over the input space
x1_range = np.linspace(np.min(X[0, :]) - 1, np.max(X[0, :]) + 1, 100)
x2_range = np.linspace(np.min(X[1, :]) - 1, np.max(X[1, :]) + 1, 100)
x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
grid_points = np.vstack([x1_grid.ravel(), x2_grid.ravel()])  # 2 x M

# Teacher predictions
Z1_t = Teacher['W1'] @ grid_points + Teacher['b1']
H1_t = relu(Z1_t)
logits_t = Teacher['W2'] @ H1_t + Teacher['b2']
probs_t = softmax(logits_t)
pred_teacher = np.argmax(probs_t, axis=0) + 1  # classes 1 or 2

# Student predictions
Z1_s = W1_student @ grid_points + b1_student
H1_s = relu(Z1_s)
logits_s = W2_student @ H1_s + b2_student
probs_s = softmax(logits_s)
pred_student = np.argmax(probs_s, axis=0) + 1

# Reshape to grid shape
pred_teacher_grid = pred_teacher.reshape(x1_grid.shape)
pred_student_grid = pred_student.reshape(x1_grid.shape)

# Plot decision boundaries
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.contourf(x1_grid, x2_grid, pred_teacher_grid, alpha=0.5, cmap='jet')
plt.scatter(X[0, :], X[1, :], c=labels, edgecolors='k')
plt.title("Teacher Decision Boundary")
plt.xlabel("Feature 1"); plt.ylabel("Feature 2")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.contourf(x1_grid, x2_grid, pred_student_grid, alpha=0.5, cmap='jet')
plt.scatter(X[0, :], X[1, :], c=labels, edgecolors='k')
plt.title("Student Decision Boundary")
plt.xlabel("Feature 1"); plt.ylabel("Feature 2")
plt.colorbar()
plt.tight_layout()
plt.show()
