#!/usr/bin/env python3
"""
adaboost_tutorial.py
--------------------
This script demonstrates AdaBoost with decision stumps on a 2D dataset.
It generates two classes of data, trains AdaBoost, and plots the decision boundary.
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

def decision_stump(X, y, D):
    """
    Finds the best decision stump.
    The stump is of the form: if feature_j < threshold then predict p, else -p.
    Returns a dictionary stump, its weighted error, and predictions.
    """
    n, d = X.shape
    best_error = np.inf
    best_stump = {'feature': None, 'threshold': None, 'polarity': 1}
    best_pred = np.zeros(n)
    
    for j in range(d):
        thresholds = np.unique(X[:, j])
        for thresh in thresholds:
            for polarity in [1, -1]:
                pred = np.ones(n)
                if polarity == 1:
                    pred[X[:, j] >= thresh] = -1
                else:
                    pred[X[:, j] >= thresh] = 1
                error = np.sum(D * (pred != y))
                if error < best_error:
                    best_error = error
                    best_stump['feature'] = j
                    best_stump['threshold'] = thresh
                    best_stump['polarity'] = polarity
                    best_pred = pred.copy()
    return best_stump, best_error, best_pred

def adaboost_train(X, y, T):
    """
    Trains AdaBoost with decision stumps.
    Returns a list of stumps and their corresponding alphas.
    """
    n = X.shape[0]
    D = np.ones(n) / n
    stumps = []
    alphas = []
    
    for t in range(T):
        stump, error, pred = decision_stump(X, y, D)
        error = max(error, 1e-10)
        alpha = 0.5 * np.log((1 - error) / error)
        stumps.append(stump)
        alphas.append(alpha)
        D = D * np.exp(-alpha * y * pred)
        D = D / np.sum(D)
    return stumps, np.array(alphas)

def adaboost_predict(X, stumps, alphas):
    """Predicts labels for X using the AdaBoost ensemble."""
    n = X.shape[0]
    agg = np.zeros(n)
    for stump, alpha in zip(stumps, alphas):
        feature = stump['feature']
        thresh = stump['threshold']
        polarity = stump['polarity']
        if polarity == 1:
            pred = np.ones(n)
            pred[X[:, feature] >= thresh] = -1
        else:
            pred = -np.ones(n)
            pred[X[:, feature] >= thresh] = 1
        agg += alpha * pred
    return np.sign(agg)

def main():
    # Generate synthetic data
    N = 100
    X1 = np.random.randn(N, 2) + 2
    X2 = np.random.randn(N, 2) - 2
    X = np.vstack((X1, X2))
    y = np.concatenate((np.ones(N), -np.ones(N)))
    
    # Train AdaBoost
    T = 50
    stumps, alphas = adaboost_train(X, y, T)
    
    # Create a grid for prediction
    x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
    y_min, y_max = X[:,1].min()-1, X[:,1].max()+1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    preds = adaboost_predict(grid_points, stumps, alphas).reshape(xx.shape)
    
    # Plot clusters and decision boundary
    plt.figure()
    plt.scatter(X[y==1,0], X[y==1,1], s=50, c='b', label='Class +1')
    plt.scatter(X[y==-1,0], X[y==-1,1], s=50, c='r', label='Class -1')
    plt.title('AdaBoost Classification - Decision Boundaries')
    plt.xlabel('Feature 1'); plt.ylabel('Feature 2')
    plt.contourf(xx, yy, preds, levels=[-np.inf, 0, np.inf], alpha=0.3,
                 colors=['lightcoral', 'lightblue'])
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
