import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

np.random.seed(42)
num_samples = 100

# Generate synthetic dataset.
X1 = np.random.randn(num_samples//2, 2) + 2   # centered at (2,2)
Y1 = np.ones((num_samples//2, 1))              # Class +1
X2 = np.random.randn(num_samples//2, 2) - 2       # centered at (-2,-2)
Y2 = -np.ones((num_samples//2, 1))             # Class -1
X = np.vstack([X1, X2])
Y = np.vstack([Y1, Y2]).ravel()

# Shuffle data.
indices = np.random.permutation(num_samples)
X = X[indices]
Y = Y[indices]

# Plot dataset.
plt.figure()
plt.scatter(X[Y==1, 0], X[Y==1, 1], color='b', label='Class +1')
plt.scatter(X[Y==-1, 0], X[Y==-1, 1], color='r', label='Class -1')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("SVM Classification Dataset")
plt.legend(loc='best')
plt.grid(True)
plt.show()

# SVM parameters.
C = 1
sigma = 0.5
# For RBF kernel in scikit-learn, gamma = 1/(2*sigma^2)
gamma_val = 1/(2*sigma**2)

# Train SVM with linear kernel.
svm_linear = SVC(C=C, kernel='linear')
svm_linear.fit(X, Y)
Y_pred_linear = svm_linear.predict(X)
accuracy_linear = accuracy_score(Y, Y_pred_linear) * 100
print(f"Linear SVM Accuracy: {accuracy_linear:.2f}%")

# Train SVM with RBF kernel.
svm_rbf = SVC(C=C, kernel='rbf', gamma=gamma_val)
svm_rbf.fit(X, Y)
Y_pred_rbf = svm_rbf.predict(X)
accuracy_rbf = accuracy_score(Y, Y_pred_rbf) * 100
print(f"RBF SVM Accuracy: {accuracy_rbf:.2f}%")

# Function to plot decision boundary.
def plot_decision_boundary(model, X, Y, title):
    x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
    y_min, y_max = X[:,1].min()-1, X[:,1].max()+1
    xx, yy = np.meshgrid(np.linspace(x_min,x_max,100), np.linspace(y_min,y_max,100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid)
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.contourf(xx, yy, Z, alpha=0.3, levels=np.linspace(Z.min(), Z.max(), 3), cmap='coolwarm')
    plt.scatter(X[Y==1, 0], X[Y==1, 1], c='b', label='Class +1')
    plt.scatter(X[Y==-1, 0], X[Y==-1, 1], c='r', label='Class -1')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

plot_decision_boundary(svm_linear, X, Y, "Linear SVM Decision Boundary")
plot_decision_boundary(svm_rbf, X, Y, "RBF SVM Decision Boundary")
