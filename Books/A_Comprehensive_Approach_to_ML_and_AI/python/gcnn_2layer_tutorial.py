#!/usr/bin/env python3
"""
gcnnTutorial_2layer.py
----------------------
A basic two-layer GCN on Zachary's Karate Club.
We use an identity feature matrix and build a GCN with one hidden layer.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def load_karate_adjacency():
    # Hard-coded Zachary Karate Club adjacency (34x34)
    # For brevity, here we load from a file or define a small sample.
    # Assume A is loaded as a NumPy array.
    A = np.loadtxt('data/karate_adjacency.txt')
    return A

def softmax_rows(X):
    expX = np.exp(X - np.max(X, axis=1, keepdims=True))
    return expX / np.sum(expX, axis=1, keepdims=True)

def cross_entropy_loss(Y_pred, Y_true, numClasses):
    # Y_true: vector of labels (1-indexed)
    m = Y_pred.shape[0]
    loss = -np.mean(np.log([Y_pred[i, int(Y_true[i])-1] for i in range(m)]))
    return loss

# Load graph
A = load_karate_adjacency()
numNodes = A.shape[0]
X = np.eye(numNodes)  # identity features
I = np.eye(numNodes)
A_tilde = A + I
D_tilde = np.diag(np.sum(A_tilde, axis=1))
D_inv_sqrt = np.diag(1/np.sqrt(np.diag(D_tilde)))
A_norm = D_inv_sqrt @ A_tilde @ D_inv_sqrt

# Use given labels (for example, 4 groups)
labels = np.zeros(numNodes)
group1 = [0,1,2,3,7,13]
group2 = [4,5,6,10,11,12,16]
group3 = [8,9,14,15,18,20,22,24,26,28,29,30,32,33]
group4 = [17,19,21,23,25,27,31]
for i in group1: labels[i] = 1
for i in group2: labels[i] = 2
for i in group3: labels[i] = 3
for i in group4: labels[i] = 4

numClasses = 4

# GCN: Two-layer
d_input = numNodes
d_hidden = 16
W1 = 0.01 * np.random.randn(d_input, d_hidden)
W2 = 0.01 * np.random.randn(d_hidden, numClasses)

# Forward pass
H1 = np.maximum(A_norm @ (X @ W1), 0)
H2 = A_norm @ (H1 @ W2)
Y_pred = softmax_rows(H2)

# Evaluate accuracy
pred_labels = np.argmax(Y_pred, axis=1) + 1
acc = np.mean(pred_labels == labels) * 100
print(f"GCN Accuracy: {acc:.2f}%")

# Plot graph with node colors by predicted label
G = nx.from_numpy_array(A)
pos = nx.spring_layout(G, seed=1)
plt.figure()
nx.draw(G, pos, node_color=pred_labels, cmap=plt.cm.jet, with_labels=True, node_size=300)
plt.title("GCN Classification on Karate Club")
plt.show()
