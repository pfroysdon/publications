#!/usr/bin/env python3
"""
gcnnTutorial_2layer_improved.py
--------------------------------
Advanced GCN on Zachary's Karate Club (Two hidden layers, partial supervision)

This script:
  1. Loads Zachary’s Karate Club graph.
  2. Defines a refined 4-group partition and sets a partial training mask.
  3. Uses identity features.
  4. Computes the normalized adjacency.
  5. Initializes a GCN with two hidden layers (each with 30 neurons) and an output layer.
  6. Trains the network for 3000 epochs using gradient descent (backpropagation through the graph convolutional layers).
  7. Every 300 epochs (or at the end) it plots the graph (using a force‐layout) colored by the predicted labels.
  
Note: In this conversion, labels and indices are converted to 0‑indexing.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def load_karate_club_adjacency():
    # Define edge list (converted from 1-indexed MATLAB to 0-indexed Python)
    edges = np.array([
        [1,2], [1,3], [2,3], [1,4], [2,4], [3,4], [1,5], [1,6], [1,7],
        [5,7], [6,7], [1,8], [2,8], [3,8], [4,8], [1,9], [3,9], [3,10],
        [1,11], [5,11], [6,11], [1,12], [1,13], [4,13], [1,14], [2,14],
        [3,14], [4,14], [6,17], [7,17], [1,18], [2,18], [1,20], [2,20],
        [1,22], [2,22], [24,26], [25,26], [3,28], [24,28], [25,28], [3,29],
        [24,30], [27,30], [2,31], [9,31], [1,32], [25,32], [26,32], [29,32],
        [3,33], [9,33], [15,33], [16,33], [19,33], [21,33], [23,33], [24,33],
        [30,33], [31,33], [32,33], [9,34], [10,34], [14,34], [15,34], [16,34],
        [19,34], [20,34], [21,34], [23,34], [24,34], [27,34], [28,34], [29,34],
        [30,34], [31,34], [32,34], [33,34]
    ])
    # Convert from 1-indexed to 0-indexed
    edges = edges - 1
    N = 34
    A = np.zeros((N, N))
    for (i, j) in edges:
        A[i, j] = 1
        A[j, i] = 1
    return A

def layout_graph(A):
    G = nx.from_numpy_array(A)
    pos = nx.spring_layout(G, seed=1)
    coords = np.array([pos[i] for i in range(len(pos))])
    return coords[:,0], coords[:,1]

def softmax_rows(X):
    ex = np.exp(X - np.max(X, axis=1, keepdims=True))
    return ex / np.sum(ex, axis=1, keepdims=True)

def cross_entropy_loss(predProbs, true_labels, numClasses):
    n = predProbs.shape[0]
    # true_labels are assumed to be 0-indexed integers.
    probs = predProbs[np.arange(n), true_labels]
    return -np.mean(np.log(probs + 1e-15))

# ---------------------- Main Script ---------------------- #
np.random.seed(1)

# 1. Load the Karate Club Graph
A = load_karate_club_adjacency()
numNodes = A.shape[0]

# 2. Refined 4-Group Labeling
# Original MATLAB groups (1-indexed):
#   group1: [1,2,3,4,8,14] -> Python indices: [0,1,2,3,7,13]
#   group2: [5,6,7,11,12,13,17] -> [4,5,6,10,11,12,16]
#   group3: [9,10,15,16,19,21,23,25,27,29,30,31,33,34] -> [8,9,14,15,18,20,22,24,26,28,29,30,32,33]
#   group4: [18,20,22,24,26,28,32] -> [17,19,21,23,25,27,31]
labels = np.zeros(numNodes, dtype=int)
group1 = np.array([0,1,2,3,7,13])
group2 = np.array([4,5,6,10,11,12,16])
group3 = np.array([8,9,14,15,18,20,22,24,26,28,29,30,32,33])
group4 = np.array([17,19,21,23,25,27,31])
labels[group1] = 0
labels[group2] = 1
labels[group3] = 2
labels[group4] = 3
numClasses = 4

# 3. Node Features: Use identity matrix (each node is one-hot)
X_features = np.eye(numNodes)
d_input = numNodes

# 4. Normalized Adjacency
I = np.eye(numNodes)
A_tilde = A + I
D_tilde = np.diag(np.sum(A_tilde, axis=1))
D_inv_sqrt = np.diag(1 / np.sqrt(np.diag(D_tilde)))
A_norm = D_inv_sqrt @ A_tilde @ D_inv_sqrt

# 5. Define Partial Training Set: label only 8 nodes (2 from each group)
train_mask = np.zeros(numNodes, dtype=bool)
# MATLAB indices: [1,2] -> Python: [0,1]; [5,6] -> [4,5]; [9,10] -> [8,9]; [18,20] -> [17,19]
train_indices = np.array([0,1,4,5,8,9,17,19])
train_mask[train_indices] = True

# 6. Initialize GCN Weights (Two hidden layers + output)
d_hidden1 = 30
d_hidden2 = 30
W1 = 0.01 * np.random.randn(d_input, d_hidden1)
W2 = 0.01 * np.random.randn(d_hidden1, d_hidden2)
W3 = 0.01 * np.random.randn(d_hidden2, numClasses)

# 7. Training Setup
learning_rate = 0.02
epochs = 3000
coordsX, coordsY = layout_graph(A)

# 8. Training Loop
loss_history = []
for ep in range(epochs):
    # Forward Pass
    Z1 = X_features @ W1         # (numNodes x d_hidden1)
    H1 = A_norm @ Z1             # (numNodes x d_hidden1)
    H1 = np.maximum(H1, 0)
    
    Z2 = H1 @ W2                 # (numNodes x d_hidden2)
    H2 = A_norm @ Z2             # (numNodes x d_hidden2)
    H2 = np.maximum(H2, 0)
    
    Z3 = H2 @ W3                 # (numNodes x numClasses)
    H3 = A_norm @ Z3             # (numNodes x numClasses)
    Y_pred = softmax_rows(H3)
    
    # Compute loss on labeled nodes and overall accuracy
    L = cross_entropy_loss(Y_pred[train_mask], labels[train_mask], numClasses)
    loss_history.append(L)
    pred_labels = np.argmax(Y_pred, axis=1)
    acc = np.mean(pred_labels == labels) * 100
    
    # Backpropagation
    dH3 = Y_pred.copy()
    labeled_idx = np.where(train_mask)[0]
    for i in labeled_idx:
        dH3[i, labels[i]] -= 1
    dH3 = dH3 / np.sum(train_mask)
    
    dZ3 = A_norm.T @ dH3
    gradW3 = H2.T @ dZ3
    
    dH2 = dZ3 @ W3.T
    dH2[H2 <= 0] = 0
    dZ2 = A_norm.T @ dH2
    gradW2 = H1.T @ dZ2
    
    dH1 = dZ2 @ W2.T
    dH1[H1 <= 0] = 0
    dZ1 = A_norm.T @ dH1
    gradW1 = X_features.T @ dZ1
    
    # Update parameters
    W3 -= learning_rate * gradW3
    W2 -= learning_rate * gradW2
    W1 -= learning_rate * gradW1
    
    if (ep % 300 == 0) or (ep == epochs - 1):
        # Visualization: Plot graph with node colors = predicted labels
        G = nx.from_numpy_array(A)
        pos = {i: (coordsX[i], coordsY[i]) for i in range(numNodes)}
        plt.figure()
        nx.draw(G, pos, with_labels=True, node_color=pred_labels, cmap=plt.cm.jet, node_size=300)
        plt.title(f"Epoch {ep+1} | Loss: {L:.3f} | Acc: {acc:.2f}%")
        plt.show()
        print(f"Epoch {ep+1} | Loss: {L:.3f} | Acc: {acc:.2f}%")

print(f"Final training accuracy: {acc:.2f}%")
