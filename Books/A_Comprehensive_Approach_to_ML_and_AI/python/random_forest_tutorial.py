import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import mode

np.random.seed(42)
num_samples = 200

# Generate synthetic dataset
X1 = np.random.randn(num_samples//2, 2) + 2   # Cluster centered at (2,2)
Y1 = np.zeros((num_samples//2, 1))              # Class 0
X2 = np.random.randn(num_samples//2, 2) - 2       # Cluster centered at (-2,-2)
Y2 = np.ones((num_samples//2, 1))               # Class 1
X = np.vstack([X1, X2])
Y = np.vstack([Y1, Y2]).ravel()

# Shuffle the data
idx = np.random.permutation(num_samples)
X = X[idx, :]
Y = Y[idx]

# Plot dataset
plt.figure()
plt.scatter(X[Y==0, 0], X[Y==0, 1], c='r', label='Class 0')
plt.scatter(X[Y==1, 0], X[Y==1, 1], c='b', label='Class 1')
plt.xlabel("Feature 1"); plt.ylabel("Feature 2")
plt.title("Random Forest Classification")
plt.legend(loc='best')
plt.grid(True)
plt.show()

# Random Forest parameters
num_trees = 20
max_depth = 4

def random_forest(X, Y, num_trees, max_depth):
    trees = []
    n_samples = X.shape[0]
    for t in range(num_trees):
        sample_idx = np.random.choice(n_samples, n_samples, replace=True)
        X_sample = X[sample_idx, :]
        Y_sample = Y[sample_idx]
        tree = DecisionTreeClassifier(max_depth=max_depth, random_state=np.random.randint(10000))
        tree.fit(X_sample, Y_sample)
        trees.append(tree)
    return trees

def predict_forest(trees, X):
    preds = np.zeros((X.shape[0], len(trees)))
    for i, tree in enumerate(trees):
        preds[:, i] = tree.predict(X)
    # Majority vote
    Y_pred, _ = mode(preds, axis=1)
    return Y_pred.ravel()

model = random_forest(X, Y, num_trees, max_depth)
print("Training complete!")

Y_pred = predict_forest(model, X)
accuracy = np.mean(Y_pred == Y) * 100
print(f"Model Accuracy: {accuracy:.2f}%")

# Decision boundary visualization
x1_range = np.linspace(np.min(X[:,0])-1, np.max(X[:,0])+1, 100)
x2_range = np.linspace(np.min(X[:,1])-1, np.max(X[:,1])+1, 100)
x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
X_grid = np.c_[x1_grid.ravel(), x2_grid.ravel()]
Y_grid = predict_forest(model, X_grid)
Y_grid = Y_grid.reshape(x1_grid.shape)

plt.figure()
plt.scatter(X[Y==0, 0], X[Y==0, 1], c='r', label='Class 0')
plt.scatter(X[Y==1, 0], X[Y==1, 1], c='b', label='Class 1')
plt.contourf(x1_grid, x2_grid, Y_grid, alpha=0.1, linewidths=0.8)
plt.xlabel("Feature 1"); plt.ylabel("Feature 2")
plt.title("Random Forest Classification - Decision Boundary")
plt.legend(loc='best')
plt.grid(True)
plt.show()
