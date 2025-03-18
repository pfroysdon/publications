import numpy as np
import matplotlib.pyplot as plt

# Environment parameters
grid_rows = 5
grid_cols = 5
num_states = grid_rows * grid_cols  # 25 states
start_state = 0   # 0-based (corresponds to MATLAB index 1)
goal_state = num_states - 1  # 24 (MATLAB index 25)
num_actions = 4  # Up, Right, Down, Left
max_steps = 50
gamma = 0.99

# Policy parameters: Theta is a (4 x 25) matrix.
Theta = np.random.randn(num_actions, num_states) * 0.01

# Hyperparameters
alpha = 0.01
num_episodes = 10000
episode_rewards = np.zeros(num_episodes)

def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

def sample_action(probs):
    r = np.random.rand()
    cumulative = np.cumsum(probs)
    return np.where(cumulative >= r)[0][0] + 1  # Return 1-indexed action

def one_hot(a, num_actions):
    vec = np.zeros(num_actions)
    vec[a-1] = 1  # a is 1-indexed
    return vec

def step(s, action, grid_rows, grid_cols, goal_state):
    # Convert state index (0-indexed) to row, col.
    row = s // grid_cols
    col = s % grid_cols
    if action == 1:       # up
        new_row = max(row - 1, 0)
        new_col = col
    elif action == 2:     # right
        new_row = row
        new_col = min(col + 1, grid_cols - 1)
    elif action == 3:     # down
        new_row = min(row + 1, grid_rows - 1)
        new_col = col
    elif action == 4:     # left
        new_row = row
        new_col = max(col - 1, 0)
    else:
        new_row, new_col = row, col
    s_next = new_row * grid_cols + new_col
    if s_next == goal_state:
        reward = 10
        done = True
    else:
        reward = -1
        done = False
    return s_next, reward, done

# Training loop using REINFORCE
for ep in range(num_episodes):
    s = start_state
    states = []
    actions = []
    rewards = []
    
    for t in range(max_steps):
        states.append(s)
        probs = softmax(Theta[:, s])
        a = sample_action(probs)  # a in {1,2,3,4} (1-indexed)
        actions.append(a)
        s_next, r, done = step(s, a, grid_rows, grid_cols, goal_state)
        rewards.append(r)
        s = s_next
        if done:
            break
    episode_rewards[ep] = np.sum(rewards)
    
    T_ep = len(rewards)
    G = np.zeros(T_ep)
    G[-1] = rewards[-1]
    for t in range(T_ep - 2, -1, -1):
        G[t] = rewards[t] + gamma * G[t+1]
    
    # Policy update
    for t in range(T_ep):
        s_t = states[t]
        a_t = actions[t]
        grad_log = one_hot(a_t, num_actions) - softmax(Theta[:, s_t])
        Theta[:, s_t] = Theta[:, s_t] + alpha * grad_log * G[t]

# Plot convergence
plt.figure()
plt.plot(np.arange(num_episodes), episode_rewards, 'b-', linewidth=1, label='Episode Reward')
window = 50
mov_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
plt.plot(np.arange(len(mov_avg)), mov_avg, 'r-', linewidth=2, label=f'Moving Average ({window} eps)')
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Policy Gradient Convergence")
plt.legend()
plt.grid(True)
plt.show()

# Evaluate learned policy (greedy)
s = start_state
optimal_path = [s]
for t in range(max_steps):
    # Greedy action: choose argmax of softmax(Theta[:, s])
    a = np.argmax(Theta[:, s]) + 1  # 1-indexed action
    s, _, done = step(s, a, grid_rows, grid_cols, goal_state)
    optimal_path.append(s)
    if done:
        break
print("Optimal path (0-indexed):", optimal_path)

# Visualize grid with greedy policy arrows.
grid = np.arange(num_states).reshape((grid_rows, grid_cols))
plt.figure()
plt.imshow(grid, cmap='gray', origin='upper')
plt.colorbar()
plt.xticks(np.arange(grid_cols), np.arange(grid_cols))
plt.yticks(np.arange(grid_rows), np.arange(grid_rows))
for s in range(num_states):
    row = s // grid_cols
    col = s % grid_cols
    p = softmax(Theta[:, s])
    bestA = np.argmax(p) + 1
    # Determine arrow direction based on bestA.
    if bestA == 1:
        dx, dy = 0, -0.4
    elif bestA == 2:
        dx, dy = 0.4, 0
    elif bestA == 3:
        dx, dy = 0, 0.4
    elif bestA == 4:
        dx, dy = -0.4, 0
    plt.arrow(col, row, dx, dy, head_width=0.2, head_length=0.2, fc='r', ec='r')
plt.title("Learned Policy on Gridworld (0-indexed States)")
plt.xlabel("Column")
plt.ylabel("Row")
plt.show()
