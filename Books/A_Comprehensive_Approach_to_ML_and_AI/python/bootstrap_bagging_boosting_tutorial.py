#!/usr/bin/env python3
"""
bootstrap_vs_bagging_vs_boosting_demo.py
----------------------------------------
This demo illustrates bootstrap, bagging, and boosting using decision stumps on a simple binary classification problem.
Synthetic 2D data is generated for two classes. A bagging ensemble and an AdaBoost ensemble are trained,
and their decision boundaries are plotted.
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

def decision_stump_train(X, y, weights):
    n, d = X.shape
    best_error = np.inf
    best_stump = {'feature': None, 'threshold': None, 'polarity': None}
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
                error = np.sum(weights * (pred != y))
                if error < best_error:
                    best_error = error
                    best_stump = {'feature': j, 'threshold': thresh, 'polarity': polarity}
                    best_pred = pred.copy()
    return best_stump, best_error, best_pred

def adaboost_train(X, y, T):
    n = X.shape[0]
    D = np.ones(n) / n
    stumps = []
    alphas = []
    for t in range(T):
        stump, error, pred = decision_stump_train(X, y, D)
        error = max(error, 1e-10)
        alpha = 0.5 * np.log((1 - error) / error)
        stump['alpha'] = alpha
        stumps.append(stump)
        alphas.append(alpha)
        D = D * np.exp(-alpha * y * pred)
        D = D / np.sum(D)
    return stumps, np.array(alphas)

def decision_stump_predict(stump, X):
    n = X.shape[0]
    predictions = np.ones(n)
    feature = stump['feature']
    thresh = stump['threshold']
    polarity = stump['polarity']
    if polarity == 1:
        predictions[X[:, feature] >= thresh] = -1
    else:
        predictions[X[:, feature] >= thresh] = 1
    return predictions

def bagging_ensemble_train(X, y, T):
    n = X.shape[0]
    models = []
    for t in range(T):
        idx = np.random.choice(n, n, replace=True)
        X_boot = X[idx]
        y_boot = y[idx]
        weights = np.ones(n) / n
        model, _, _ = decision_stump_train(X_boot, y_boot, weights)
        models.append(model)
    return models

def bagging_ensemble_predict(models, X):
    n = X.shape[0]
    T = len(models)
    pred_matrix = np.zeros((n, T))
    for t, model in enumerate(models):
        pred_matrix[:, t] = decision_stump_predict(model, X)
    predictions = np.sign(np.sum(pred_matrix, axis=1))
    predictions[predictions == 0] = 1
    return predictions

def adaboost_predict(X, stumps, alphas):
    n = X.shape[0]
    agg = np.zeros(n)
    for stump, alpha in zip(stumps, alphas):
        pred = decision_stump_predict(stump, X)
        agg += alpha * pred
    predictions = np.sign(agg)
    predictions[predictions == 0] = 1
    return predictions

def main():
    # Generate synthetic data
    N = 200
    X1 = np.random.randn(100, 2) + np.array([-1, -1])
    X2 = np.random.randn(100, 2) + np.array([1, 1])
    X = np.vstack((X1, X2))
    y = np.concatenate(( -np.ones(100), np.ones(100) ))
    
    T = 50
    bagging_models = bagging_ensemble_train(X, y, T)
    y_pred_bagging = bagging_ensemble_predict(bagging_models, X)
    acc_bagging = np.mean(y_pred_bagging == y)
    print(f"Bagging Training Accuracy: {acc_bagging*100:.2f}%")
    
    stumps, alphas = adaboost_train(X, y, T)
    y_pred_boosting = adaboost_predict(X, stumps, alphas)
    acc_boosting = np.mean(y_pred_boosting == y)
    print(f"Boosting Training Accuracy: {acc_boosting*100:.2f}%")
    
    # Create grid for decision boundary
    x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
    y_min, y_max = X[:,1].min()-1, X[:,1].max()+1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    preds_bagging = bagging_ensemble_predict(bagging_models, grid_points).reshape(xx.shape)
    preds_boosting = adaboost_predict(grid_points, stumps, alphas).reshape(xx.shape)
    
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.contourf(xx, yy, preds_bagging, levels=[-np.inf, 0, np.inf], alpha=0.3, colors=['lightcoral', 'lightblue'])
    plt.scatter(X[y==-1, 0], X[y==-1, 1], color='r', label="Class -1")
    plt.scatter(X[y==1, 0], X[y==1, 1], color='b', label="Class +1")
    plt.title(f"Bagging Decision Boundary (T={T})")
    plt.xlabel("x1"); plt.ylabel("x2")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1,2,2)
    plt.contourf(xx, yy, preds_boosting, levels=[-np.inf, 0, np.inf], alpha=0.3, colors=['lightcoral', 'lightblue'])
    plt.scatter(X[y==-1, 0], X[y==-1, 1], color='r', label="Class -1")
    plt.scatter(X[y==1, 0], X[y==1, 1], color='b', label="Class +1")
    plt.title(f"Boosting Decision Boundary (T={T})")
    plt.xlabel("x1"); plt.ylabel("x2")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
