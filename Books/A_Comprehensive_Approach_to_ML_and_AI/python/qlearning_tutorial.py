import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Define the reward matrix R (6x6) using 0-based indexing.
R = np.array([
    [-1, -1, -1, -1,  0, -1],
    [-1, -1, -1,  0, -1, 100],
    [-1, -1, -1,  0, -1, -1],
    [-1,  0,  0, -1,  0, -1],
    [ 0, -1, -1,  0, -1, 100],
    [-1,  0, -1, -1,  0, 100]
])
num_states = 6
start_state = 2  # 0-based state 2
goal_state = 5   # 0-based state 5

alpha = 0.1
gamma = 0.8
epsilon = 0.1
num_episodes = 400
Q = np.zeros((num_states, num_states))
Q_norm_history = np.zeros(num_episodes)

for ep in range(num_episodes):
    s = start_state
    while s != goal_state:
        valid_actions = np.where(R[s, :] != -1)[0]  # actions available from state s (0-based)
        if valid_actions.size == 0:
            break
        if np.random.rand() < epsilon:
            a = np.random.choice(valid_actions)
        else:
            Q_valid = Q[s, valid_actions]
            a = valid_actions[np.argmax(Q_valid)]
        s_next = a
        reward = R[s, a]
        if s_next == goal_state:
            Q[s, a] = Q[s, a] + alpha * (reward - Q[s, a])
            break
        else:
            Q[s, a] = Q[s, a] + alpha * (reward + gamma * np.max(Q[s_next, :]) - Q[s, a])
        s = s_next
    Q_norm_history[ep] = np.linalg.norm(Q)

print("Learned Q-table:")
print(Q)

# Plot norm of Q per episode
plt.figure()
plt.plot(Q_norm_history, 'b-', linewidth=2)
plt.xlabel("Episode")
plt.ylabel("||Q||_2")
plt.title("Q-learning Convergence")
plt.grid(True)
plt.show()

# Extract the optimal path starting from state 2.
optimal_path = [start_state]
s = start_state
while s != goal_state:
    a = np.argmax(Q[s, :])
    optimal_path.append(a)
    s = a
print("Optimal path (0-indexed):", optimal_path)

# Build directed graph from R.
edges_from = []
edges_to = []
edge_weights = []
for i in range(num_states):
    for j in range(num_states):
        if R[i, j] != -1:
            edges_from.append(i)
            edges_to.append(j)
            edge_weights.append(R[i, j])

G = nx.DiGraph()
for u, v, w in zip(edges_from, edges_to, edge_weights):
    G.add_edge(u, v, weight=w)

# Plot graph with 0-based labels and highlight the optimal path.
pos = nx.spring_layout(G)
plt.figure()
nx.draw(G, pos, with_labels=True, node_color='lightblue', arrows=True)
# Highlight optimal path in red.
optimal_edges = [(optimal_path[i], optimal_path[i+1]) for i in range(len(optimal_path)-1)]
nx.draw_networkx_edges(G, pos, edgelist=optimal_edges, edge_color='r', width=2)
plt.title("Directed Graph with Optimal Path")
plt.show()

# Also, visualize Q-table with imagesc equivalent.
plt.figure()
plt.imshow(Q, cmap='jet', interpolation='nearest')
plt.colorbar()
plt.xlabel("Next State (0-indexed)")
plt.ylabel("Current State (0-indexed)")
plt.title("Learned Q-values Matrix")
for i in range(num_states):
    for j in range(num_states):
        if (i, j) in optimal_edges:
            plt.plot(j, i, 'ko', markersize=10, markeredgewidth=2)
plt.show()
