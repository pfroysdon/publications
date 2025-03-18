#!/usr/bin/env python3
"""
decisionTreeTutorial_tree.py
----------------------------
This script loads a dataset from 'data/data.txt', builds a decision tree (using the functions defined below),
and visualizes the tree structure using NetworkX.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def majority_vote(y):
    return np.round(np.mean(y)).astype(int)

def ent(y):
    # Compute entropy in base 2
    values, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return -np.sum(p * np.log2(p + 1e-10))

def cond_ent(y, X_binary):
    # Conditional entropy of y given binary split X_binary
    vals = np.unique(X_binary)
    ce = 0
    for v in vals:
        idx = (X_binary == v)
        ce += (np.sum(idx)/len(y)) * ent(y[idx])
    return ce

def find_best_split(X, y):
    n, d = X.shape
    best_ig = -np.inf
    best_feature, best_threshold, best_split = None, None, None
    H = ent(y)
    for j in range(d):
        values = np.unique(X[:, j])
        if len(values) < 2:
            continue
        thresholds = 0.5 * (values[:-1] + values[1:])
        for thresh in thresholds:
            split = X[:, j] < thresh
            if split.sum() == 0 or (~split).sum() == 0:
                continue
            H_cond = cond_ent(y, split.astype(int))
            IG = H - H_cond
            if IG > best_ig:
                best_ig = IG
                best_feature = j
                best_threshold = thresh
                best_split = split
    return best_feature, best_threshold, best_split

def build_tree(X, y, cols, depth=0, max_depth=5):
    if (np.all(y == y[0])) or (depth >= max_depth) or (X.shape[0] < 2):
        return {'isLeaf': True, 'prediction': majority_vote(y), 'indices': np.arange(X.shape[0])}
    best_feature, best_threshold, split = find_best_split(X, y)
    if best_feature is None:
        return {'isLeaf': True, 'prediction': majority_vote(y), 'indices': np.arange(X.shape[0])}
    left_tree = build_tree(X[split], y[split], cols, depth+1, max_depth)
    right_tree = build_tree(X[~split], y[~split], cols, depth+1, max_depth)
    return {'isLeaf': False,
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_tree,
            'right': right_tree,
            'label': f"{cols[best_feature]} < {best_threshold:.2f}"}

def traverse_tree_collect(tree, G, parent=None, node_id=0, pos_dict=None, labels=None):
    # Recursively collect nodes and edges for visualization.
    if pos_dict is None:
        pos_dict = {}
    if labels is None:
        labels = {}
    current_id = node_id
    if tree['isLeaf']:
        labels[current_id] = f"Leaf: {tree['prediction']}"
    else:
        labels[current_id] = tree['label']
    pos_dict[current_id] = (depth_to_x(tree), -current_id)  # simple positioning
    if parent is not None:
        G.add_edge(parent, current_id)
    next_id = current_id + 1
    if not tree['isLeaf']:
        G, next_id, pos_dict, labels = traverse_tree_collect(tree['left'], G, current_id, next_id, pos_dict, labels)
        G, next_id, pos_dict, labels = traverse_tree_collect(tree['right'], G, current_id, next_id, pos_dict, labels)
    return G, next_id, pos_dict, labels

def depth_to_x(tree):
    # A simple function to determine horizontal position (for visualization)
    if tree['isLeaf']:
        return 0
    else:
        return 1

# Load data
M = np.loadtxt('data/data.txt', delimiter='\t')
Y = M[:,0]
X = M[:,1:]
cols = ['cyl', 'dis', 'hp', 'wgt', 'acc', 'mtn', 'mkr']

tree = build_tree(X, Y, cols)

# Visualize the tree using networkx
G = nx.DiGraph()
G, _, pos_dict, labels = traverse_tree_collect(tree, G)
plt.figure(figsize=(8,6))
nx.draw(G, pos=pos_dict, labels=labels, with_labels=True, node_size=1500, node_color='lightblue', arrows=True)
plt.title("Decision Tree")
plt.show()
