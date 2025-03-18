#!/usr/bin/env python3
"""
gcnnTutorial_4layer.py
-----------------------
Advanced GCN on Zachary's Karate Club with 4 Hidden Layers

This script implements a GCN with four hidden layers (plus an output layer) for node classification.
We use a refined 4‑group partition and full supervision (all nodes labeled). The network architecture is:

   H1 = ReLU( A_norm @ (X @ W1) )
   H2 = ReLU( A_norm @ (H1 @ W2) )
   H3 = ReLU( A_norm @ (H2 @ W3) )
   H4 = ReLU( A_norm @ (H3 @ W4) )
   H5 = A_norm @ (H4 @ W5)   --> row-wise softmax for class probabilities

The network is trained for 3000 epochs with a learning rate of 0.02.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def load_karate_club_adjacency():
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
    edges = edges - 1  # convert to 0-indexed
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
    probs = predProbs[np.arange(n), true_labels]
    return -np.mean(np.log(probs + 1e-15))

# ---------------------- Main Script ---------------------- #
np.random.seed(1)

# 1. Load the Karate Club Graph
A = load_karate_club_adjacency()
numNodes = A.shape[0]

# 2. Refined 4-Group Labeling
# Group definitions (0-indexed):
#   group1: [0,1,2,3,7,13]
#   group2: [4,5,6,10,11,12,16]
#   group3: [8,9,14,15,18,20,22,24,26,28,29,30,32,33]
#   group4: [17,19,21,23,25,27,31]
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

# 3. Node Features: Identity matrix
X_features = np.eye(numNodes)
d_input = numNodes

# 4. Normalized Adjacency
I = np.eye(numNodes)
A_tilde = A + I
D_tilde = np.diag(np.sum(A_tilde, axis=1))
D_inv_sqrt = np.diag(1 / np.sqrt(np.diag(D_tilde)))
A_norm = D_inv_sqrt @ A_tilde @ D_inv_sqrt

# 5. Full Supervision: all nodes labeled
train_mask = np.ones(numNodes, dtype=bool)

# 6. Initialize GCN Weights: 4 hidden layers
d_hidden1 = 32
d_hidden2 = 32
d_hidden3 = 32
d_hidden4 = 32
W1 = 0.01 * np.random.randn(d_input, d_hidden1)
W2 = 0.01 * np.random.randn(d_hidden1, d_hidden2)
W3 = 0.01 * np.random.randn(d_hidden2, d_hidden3)
W4 = 0.01 * np.random.randn(d_hidden3, d_hidden4)
W5 = 0.01 * np.random.randn(d_hidden4, numClasses)

# 7. Training Setup
learning_rate = 0.02
epochs = 3000
coordsX, coordsY = layout_graph(A)

# 8. Training Loop
loss_history = []
for ep in range(epochs):
    # Layer 1
    Z1 = X_features @ W1
    H1 = A_norm @ Z1
    H1 = np.maximum(H1, 0)
    # Layer 2
    Z2 = H1 @ W2
    H2 = A_norm @ Z2
    H2 = np.maximum(H2, 0)
    # Layer 3
    Z3 = H2 @ W3
    H3 = A_norm @ Z3
    H3 = np.maximum(H3, 0)
    # Layer 4
    Z4 = H3 @ W4
    H4 = A_norm @ Z4
    H4 = np.maximum(H4, 0)
    # Output layer
    Z5 = H4 @ W5
    H5 = A_norm @ Z5
    Y_pred = softmax_rows(H5)
    
    L = cross_entropy_loss(Y_pred[train_mask], labels[train_mask], numClasses)
    loss_history.append(L)
    pred_labels = np.argmax(Y_pred, axis=1)
    acc = np.mean(pred_labels == labels) * 100
    
    # Backpropagation:
    dH5 = Y_pred.copy()
    for i in range(numNodes):
        dH5[i, labels[i]] -= 1
    dH5 = dH5 / numNodes
    
    dZ5 = A_norm.T @ dH5
    gradW5 = H4.T @ dZ5
    
    dH4 = dZ5 @ W5.T
    dH4[H4 <= 0] = 0
    dZ4 = A_norm.T @ dH4
    gradW4 = H3.T @ dZ4
    
    dH3 = dZ4 @ W4.T
    dH3[H3 <= 0] = 0
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
    W5 -= learning_rate * gradW5
    W4 -= learning_rate * gradW4
    W3 -= learning_rate * gradW3
    W2 -= learning_rate * gradW2
    W1 -= learning_rate * gradW1
    
    if (ep % 200 == 0) or (ep == epochs - 1):
        plt.figure()
        G = nx.from_numpy_array(A)
        pos = {i: (coordsX[i], coordsY[i]) for i in range(numNodes)}
        nx.draw(G, pos, with_labels=True, node_color=pred_labels, cmap=plt.cm.jet, node_size=300)
        plt.title(f"Epoch {ep+1} | Loss: {L:.3f} | Acc: {acc:.2f}%")
        plt.show()
        print(f"Epoch {ep+1} | Loss: {L:.3f} | Acc: {acc:.2f}%")
        
print(f"Final training accuracy: {acc:.2f}%")
