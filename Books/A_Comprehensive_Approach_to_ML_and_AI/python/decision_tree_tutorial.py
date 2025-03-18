#!/usr/bin/env python3
"""
decisionTreeTutorial.py
-------------------------
This tutorial demonstrates a decision tree classifier built entirely from scratch.
It generates a 2D dataset with two classes, recursively builds a tree using the Gini impurity,
and visualizes the decision boundary.
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

# 1. Generate Synthetic Data
N = 100
# Class +1: centered at (2,2)
X1 = np.random.randn(N, 2) + 2
# Class -1: centered at (-2,-2)
X2 = np.random.randn(N, 2) - 2
X = np.vstack((X1, X2))
y = np.concatenate((np.ones(N), -np.ones(N)))

# 2. Build Decision Tree from Scratch

def majority_vote(y):
    return 1 if np.sum(y == 1) >= np.sum(y == -1) else -1

def gini_impurity(y):
    n = len(y)
    if n == 0:
        return 0
    p = np.sum(y == 1) / n
    return 1 - (p**2 + (1-p)**2)

def find_best_split(X, y):
    n, d = X.shape
    best_impurity = np.inf
    best_feature, best_threshold, best_split = None, None, None
    for j in range(d):
        values = np.unique(X[:, j])
        if len(values) < 2:
            continue
        thresholds = 0.5 * (values[:-1] + values[1:])
        for thresh in thresholds:
            split = X[:, j] < thresh
            if split.sum() == 0 or (~split).sum() == 0:
                continue
            impurity_left = gini_impurity(y[split])
            impurity_right = gini_impurity(y[~split])
            weighted_impurity = (split.sum()/n)*impurity_left + ((~split).sum()/n)*impurity_right
            if weighted_impurity < best_impurity:
                best_impurity = weighted_impurity
                best_feature = j
                best_threshold = thresh
                best_split = split
    return best_feature, best_threshold, best_impurity, best_split

def build_tree(X, y, depth, max_depth):
    # stopping criteria
    if (np.all(y == y[0])) or (depth >= max_depth) or (X.shape[0] < 2):
        return {'is_leaf': True, 'prediction': majority_vote(y)}
    best_feature, best_threshold, best_impurity, split = find_best_split(X, y)
    if best_feature is None:
        return {'is_leaf': True, 'prediction': majority_vote(y)}
    # If one side is empty, make leaf
    if split.sum() == 0 or (~split).sum() == 0:
        return {'is_leaf': True, 'prediction': majority_vote(y)}
    left_tree = build_tree(X[split], y[split], depth+1, max_depth)
    right_tree = build_tree(X[~split], y[~split], depth+1, max_depth)
    return {'is_leaf': False,
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_tree,
            'right': right_tree}

def predict_tree(tree, X):
    n = X.shape[0]
    preds = np.zeros(n)
    for i in range(n):
        preds[i] = traverse_tree(tree, X[i, :])
    return preds

def traverse_tree(tree, x):
    if tree['is_leaf']:
        return tree['prediction']
    if x[tree['feature']] < tree['threshold']:
        return traverse_tree(tree['left'], x)
    else:
        return traverse_tree(tree['right'], x)

max_depth = 3
tree = build_tree(X, y, depth=0, max_depth=max_depth)

# 3. Visualize Decision Boundary
x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
y_min, y_max = X[:,1].min()-1, X[:,1].max()+1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))
grid_points = np.c_[xx.ravel(), yy.ravel()]
preds = predict_tree(tree, grid_points).reshape(xx.shape)

plt.figure(figsize=(8,6))
plt.contourf(xx, yy, preds, levels=[-np.inf, 0, np.inf], alpha=0.3, colors=['lightcoral', 'lightblue'])
plt.scatter(X[y==1,0], X[y==1,1], c='b', marker='o', label='Class +1')
plt.scatter(X[y==-1,0], X[y==-1,1], c='r', marker='o', label='Class -1')
plt.title("Decision Tree Classification")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend(loc='upper right')
plt.grid(True)
plt.show()
