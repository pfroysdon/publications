import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
N = 100
# Class 1: centered at (2,2)
X1 = np.random.randn(N, 2) + 2
# Class 0: centered at (-2,-2)
X0 = np.random.randn(N, 2) - 2
X = np.vstack([X1, X0])
y = np.hstack([np.ones(N), np.zeros(N)])  # labels in {0,1}

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def stump_reg_train(X, r):
    n, d = X.shape
    best_loss = np.inf
    best_feature = 0
    best_threshold = 0
    best_c1 = 0
    best_c2 = 0
    for j in range(d):
        xj = X[:, j]
        unique_vals = np.unique(xj)
        for thresh in unique_vals:
            left_idx = xj < thresh
            right_idx = xj >= thresh
            if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
                continue
            c1 = np.mean(r[left_idx])
            c2 = np.mean(r[right_idx])
            loss_left = np.sum((r[left_idx] - c1)**2)
            loss_right = np.sum((r[right_idx] - c2)**2)
            loss = loss_left + loss_right
            if loss < best_loss:
                best_loss = loss
                best_feature = j
                best_threshold = thresh
                best_c1 = c1
                best_c2 = c2
    stump = {'feature': best_feature, 'threshold': best_threshold, 'c1': best_c1, 'c2': best_c2}
    return stump

def stump_reg_predict(stump, X):
    j = stump['feature']
    thresh = stump['threshold']
    c1 = stump['c1']
    c2 = stump['c2']
    preds = np.where(X[:, j] < thresh, c1, c2)
    return preds

def xgboost_train(X, y, T, eta):
    n = X.shape[0]
    F = np.zeros(n)
    models = []
    for t in range(T):
        p = sigmoid(F)
        grad = y - p  # negative gradient: residuals we want to fit.
        stump = stump_reg_train(X, grad)
        h = stump_reg_predict(stump, X)
        alpha = np.dot(grad, h) / (np.dot(h, h) + 1e-12)
        F = F + eta * alpha * h
        models.append({'stump': stump, 'coef': eta * alpha})
    return models

def xgboost_predict(X, models):
    n = X.shape[0]
    F = np.zeros(n)
    for model in models:
        stump = model['stump']
        coef = model['coef']
        F += coef * stump_reg_predict(stump, X)
    p = sigmoid(F)
    return p

T = 50
eta = 0.1
models = xgboost_train(X, y, T, eta)
preds = xgboost_predict(X, models)
accuracy = np.mean((preds >= 0.5).astype(int) == y) * 100
print(f"XGBoost-like Model Accuracy: {accuracy:.2f}%")

# Visualize decision boundary.
x1_min, x1_max = X[:,0].min()-1, X[:,0].max()+1
x2_min, x2_max = X[:,1].min()-1, X[:,1].max()+1
xx, yy = np.meshgrid(np.linspace(x1_min, x1_max, 200), np.linspace(x2_min, x2_max, 200))
grid_points = np.c_[xx.ravel(), yy.ravel()]
grid_preds = xgboost_predict(grid_points, models)
grid_preds = grid_preds.reshape(xx.shape)

plt.figure()
plt.scatter(X[y==1,0], X[y==1,1], color='b', label='Class 1')
plt.scatter(X[y==0,0], X[y==0,1], color='r', label='Class 0')
plt.contourf(xx, yy, grid_preds, levels=[-0.5, 0.5, 1.5], alpha=0.1, colors=['r','b'])
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("XGBoost-like Decision Boundary")
plt.legend()
plt.grid(True)
plt.show()
