import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

# Environment parameters
state_min = 0
state_max = 10
start_state = 0
goal_state = 10

def human_reward(s):
    return s  # human reward is simply the state value

# Reward model: r_model(s) = theta0 + theta1 * s, initialize theta as zeros.
theta = np.array([0.0, 0.0])  # shape (2,)

# Policy parameters: logistic policy: p(right|s) = sigmoid(phi * s + b)
phi = 0.0
b = 0.0

# Hyperparameters
num_episodes = 1000
max_steps = 20
alpha_policy = 0.01
alpha_model = 0.001
gamma = 0.99

# Memory for reward model training
model_states = []
model_rewards = []

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

episode_rewards = np.zeros(num_episodes)

# Training loop
for ep in range(num_episodes):
    s = start_state
    states_ep = []
    actions_ep = []
    rewards_ep = []
    # Generate an episode
    for t in range(max_steps):
        states_ep.append(s)
        p_right = sigmoid(phi * s + b)
        a = 1 if np.random.rand() < p_right else -1
        actions_ep.append(a)
        s_next = np.clip(s + a, state_min, state_max)
        r = human_reward(s)
        rewards_ep.append(r)
        s = s_next
        if s == goal_state:
            break

    T_ep = len(rewards_ep)
    # Compute discounted returns.
    G = np.zeros(T_ep)
    G[-1] = rewards_ep[-1]
    for t in range(T_ep-2, -1, -1):
        G[t] = rewards_ep[t] + gamma * G[t+1]
    episode_rewards[ep] = np.sum(rewards_ep)
    
    # Collect data for reward model training.
    model_states.extend(states_ep)
    model_rewards.extend([human_reward(s_val) for s_val in states_ep])
    
    # Update reward model via one-step gradient descent on MSE.
    # Our model predicts: r_model(s) = theta0 + theta1 * s.
    model_states_arr = np.array(model_states)  # shape (N,)
    model_rewards_arr = np.array(model_rewards)  # shape (N,)
    predictions = theta[0] + theta[1] * model_states_arr
    error = predictions - model_rewards_arr
    grad_theta0 = np.mean(error)
    grad_theta1 = np.mean(error * model_states_arr)
    theta = theta - alpha_model * np.array([grad_theta0, grad_theta1])
    
    # Update policy using REINFORCE with reward model as reward signal.
    r_model_ep = theta[0] + theta[1] * np.array(states_ep)
    baseline = np.mean(G)
    for t in range(T_ep):
        s_t = states_ep[t]
        p_right = sigmoid(phi * s_t + b)
        grad_log = (1 - p_right) if actions_ep[t] == 1 else (0 - p_right)
        advantage = G[t] - baseline
        phi = phi + alpha_policy * grad_log * s_t * advantage
        b = b + alpha_policy * grad_log * advantage

    if (ep+1) % 100 == 0:
        print(f"Episode {ep+1}, Total Reward: {np.sum(rewards_ep):.2f}")

# Evaluate learned policy via a greedy episode.
s = start_state
trajectory = [s]
for t in range(max_steps):
    p_right = sigmoid(phi * s + b)
    a = 1 if p_right >= 0.5 else -1
    s = np.clip(s + a, state_min, state_max)
    trajectory.append(s)
    if s == goal_state:
        break

# Plot the state trajectory.
plt.figure()
plt.plot(range(len(trajectory)), trajectory, '-o', linewidth=2)
plt.xlabel("Time Step")
plt.ylabel("State")
plt.title("Agent Trajectory under Learned Policy")
plt.grid(True)
plt.show()

# Plot the learned reward model vs. the human reward.
s_vals = np.linspace(state_min, state_max, 100)
r_model_vals = theta[0] + theta[1] * s_vals
r_human_vals = s_vals  # since human_reward(s)=s
plt.figure()
plt.plot(s_vals, r_model_vals, 'r-', linewidth=2, label="Learned Reward Model")
plt.plot(s_vals, r_human_vals, 'b--', linewidth=2, label="Human Reward")
plt.xlabel("State")
plt.ylabel("Reward")
plt.legend()
plt.title("Reward Model vs. Human Reward")
plt.grid(True)
plt.show()
