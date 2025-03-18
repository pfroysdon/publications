#!/usr/bin/env python3
"""
grpoTutorial_pendulum.py
------------------------
A simplified Group Relative Policy Optimization (GRPO) implementation for an inverted pendulum.
This script defines a custom inverted pendulum environment, a simple actor and critic,
collects an episode, computes group‐based policy loss, and updates parameters via finite‐difference gradients.
Note: Finite-difference gradient estimation is slow and used here for educational purposes.
"""

import numpy as np
import matplotlib.pyplot as plt

# --- Inverted Pendulum Environment ---
class InvertedPendulum:
    def __init__(self):
        self.dt = 0.02
        self.maxEpisodeSteps = 200
        self.maxTorque = 2.0
        self.reset()
    
    def reset(self):
        # State: [theta, theta_dot]; observation: [cos(theta), sin(theta), theta_dot]
        self.theta = np.random.uniform(-0.1, 0.1)
        self.theta_dot = 0.0
        self.steps = 0
        return self._get_obs()
    
    def step(self, torque):
        self.steps += 1
        g = 9.8; l = 1.0; m = 1.0
        theta_acc = (g / l) * np.sin(self.theta) + torque / (m * l**2)
        self.theta_dot += theta_acc * self.dt
        self.theta += self.theta_dot * self.dt
        reward = -abs(self.theta)
        done = (self.steps >= self.maxEpisodeSteps) or (abs(self.theta) > np.pi/2)
        return self._get_obs(), reward, done
    
    def _get_obs(self):
        return np.array([np.cos(self.theta), np.sin(self.theta), self.theta_dot])
    
    @property
    def state(self):
        return np.array([self.theta, self.theta_dot])

# --- Actor and Critic Functions ---
def actor_forward(actor, s):
    # s: observation vector (3,)
    z1 = actor['W1'] @ s + actor['b1']
    h1 = np.maximum(0, z1)  # ReLU
    mean_a = actor['W2'] @ h1 + actor['b2']
    std_a = np.exp(actor['log_std'])
    return mean_a, std_a

def gaussian_log_prob(a, mean, std):
    # Returns log probability (scalar) under Gaussian
    logp = -0.5 * np.log(2*np.pi) - np.log(std + 1e-8) - 0.5 * ((a - mean)/std)**2
    return np.sum(logp)

def finite_diff_gradient(param, loss_func, delta):
    grad = np.zeros_like(param)
    it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        orig = param[idx]
        param[idx] = orig + delta
        loss_plus = loss_func(param)
        param[idx] = orig - delta
        loss_minus = loss_func(param)
        grad[idx] = (loss_plus - loss_minus) / (2 * delta)
        param[idx] = orig
        it.iternext()
    return grad

def grpo_loss(actor, obs_batch, actions_batch, returns, logprobs_batch, mu_good, mu_bad):
    # Compute GRPO loss over a trajectory.
    T = obs_batch.shape[1]
    loss_sum = 0
    for i in range(T):
        s = obs_batch[:, i]
        a = actions_batch[:, i]
        mean_a, std_a = actor_forward(actor, s)
        pred_logp = gaussian_log_prob(a, mean_a, std_a)
        if returns[i] >= mu_good:
            loss_sum -= pred_logp * (returns[i] - mu_good)
        else:
            loss_sum += pred_logp * (mu_bad - returns[i])
    return loss_sum / T

def update_actor_grpo(actor, obs_batch, actions_batch, returns, logprobs_batch, mu_good, mu_bad, actorLR, delta):
    # Update each actor parameter using finite-difference gradients.
    actor['W1'] -= actorLR * finite_diff_gradient(actor['W1'], lambda p: grpo_loss({**actor, 'W1': p}, obs_batch, actions_batch, returns, logprobs_batch, mu_good, mu_bad), delta)
    actor['b1'] -= actorLR * finite_diff_gradient(actor['b1'], lambda p: grpo_loss({**actor, 'b1': p}, obs_batch, actions_batch, returns, logprobs_batch, mu_good, mu_bad), delta)
    actor['W2'] -= actorLR * finite_diff_gradient(actor['W2'], lambda p: grpo_loss({**actor, 'W2': p}, obs_batch, actions_batch, returns, logprobs_batch, mu_good, mu_bad), delta)
    actor['b2'] -= actorLR * finite_diff_gradient(actor['b2'], lambda p: grpo_loss({**actor, 'b2': p}, obs_batch, actions_batch, returns, logprobs_batch, mu_good, mu_bad), delta)
    actor['log_std'] -= actorLR * finite_diff_gradient(actor['log_std'], lambda p: grpo_loss({**actor, 'log_std': p}, obs_batch, actions_batch, returns, logprobs_batch, mu_good, mu_bad), delta)
    return actor

def critic_forward(critic, s):
    z1 = critic['W1'] @ s + critic['b1']
    h1 = np.maximum(0, z1)
    Q = critic['W2'] @ h1 + critic['b2']
    return Q

def critic_loss(critic, obs_batch, returns):
    T = obs_batch.shape[1]
    loss_sum = 0
    for i in range(T):
        s = obs_batch[:, i]
        Q_pred = critic_forward(critic, s)
        loss_sum += (Q_pred - returns[i])**2
    return loss_sum / T

def update_critic(critic, obs_batch, returns, criticLR, delta):
    critic['W1'] -= criticLR * finite_diff_gradient(critic['W1'], lambda p: critic_loss({**critic, 'W1': p}, obs_batch, returns), delta)
    critic['b1'] -= criticLR * finite_diff_gradient(critic['b1'], lambda p: critic_loss({**critic, 'b1': p}, obs_batch, returns), delta)
    critic['W2'] -= criticLR * finite_diff_gradient(critic['W2'], lambda p: critic_loss({**critic, 'W2': p}, obs_batch, returns), delta)
    critic['b2'] -= criticLR * finite_diff_gradient(critic['b2'], lambda p: critic_loss({**critic, 'b2': p}, obs_batch, returns), delta)
    return critic

# --- Main GRPO Script ---
env = InvertedPendulum()

numEpisodes = 150
gamma = 0.99
actorLR = 1e-3
criticLR = 1e-3
explorationNoise = 0.1
delta = 1e-5

obs_dim = 3
action_dim = 1
hidden_size = 16

# Initialize Actor
actor = {
    'W1': 0.1 * np.random.randn(hidden_size, obs_dim),
    'b1': np.zeros((hidden_size, 1)),
    'W2': 0.1 * np.random.randn(action_dim, hidden_size),
    'b2': np.zeros((action_dim, 1)),
    'log_std': -0.5 * np.ones((action_dim, 1))
}

# Initialize Critic
critic = {
    'W1': 0.1 * np.random.randn(hidden_size, obs_dim),
    'b1': np.zeros((hidden_size, 1)),
    'W2': 0.1 * np.random.randn(1, hidden_size),
    'b2': 0.0
}

episodeRewards = []

for ep in range(numEpisodes):
    s = env.reset()
    traj_obs = []
    traj_actions = []
    traj_rewards = []
    traj_logp = []
    
    done = False
    while not done:
        s = s.reshape(-1,1)
        mean_a, std_a = actor_forward(actor, s)
        a = mean_a + std_a * np.random.randn(*std_a.shape)
        a = np.clip(a, -env.maxTorque, env.maxTorque)
        logp = gaussian_log_prob(a, mean_a, std_a)
        
        traj_obs.append(s.flatten())
        traj_actions.append(a.flatten())
        
        s_next, r, done = env.step(a)
        traj_rewards.append(r)
        traj_logp.append(logp)
        s = s_next
    traj_obs = np.array(traj_obs).T  # shape: (obs_dim, T)
    traj_actions = np.array(traj_actions).T  # shape: (action_dim, T)
    traj_rewards = np.array(traj_rewards)
    traj_logp = np.array(traj_logp)
    
    # Compute returns
    T_steps = len(traj_rewards)
    returns = np.zeros(T_steps)
    G = 0
    for t in range(T_steps-1, -1, -1):
        G = traj_rewards[t] + gamma * G
        returns[t] = G
    
    episodeRewards.append(np.sum(traj_rewards))
    
    # Grouping: compute median return and average returns for good and bad groups
    med_return = np.median(returns)
    good_idx = returns >= med_return
    bad_idx = returns < med_return
    mu_good = np.mean(returns[good_idx]) if np.sum(good_idx) > 0 else med_return
    mu_bad = np.mean(returns[bad_idx]) if np.sum(bad_idx) > 0 else med_return
    
    # Compute GRPO loss (for reporting)
    loss_grpo = grpo_loss(actor, traj_obs, traj_actions, returns, traj_logp, mu_good, mu_bad)
    
    # Update actor and critic using finite differences
    actor = update_actor_grpo(actor, traj_obs, traj_actions, returns, traj_logp, mu_good, mu_bad, actorLR, delta)
    critic = update_critic(critic, traj_obs, returns, criticLR, delta)
    
    print(f"Episode {ep+1}, Total Reward: {episodeRewards[-1]:.2f}, GRPO Loss: {loss_grpo:.4f}")

# Plot convergence of episode rewards
plt.figure()
plt.plot(np.arange(1, numEpisodes+1), episodeRewards, 'b-', linewidth=1)
plt.xlabel("Episode"); plt.ylabel("Total Reward")
plt.title("GRPO Convergence on Inverted Pendulum")
plt.grid(True)
plt.show()

# Final Test Run: record trajectory using deterministic policy
s = env.reset()
trajectory = []
done = False
while not done:
    s = s.reshape(-1,1)
    mean_a, _ = actor_forward(actor, s)
    a = np.clip(mean_a, -env.maxTorque, env.maxTorque)
    s, _, done = env.step(a)
    trajectory.append(env.state.copy())
trajectory = np.array(trajectory)
time = np.arange(trajectory.shape[0]) * env.dt

plt.figure()
plt.subplot(2,1,1)
plt.plot(time, trajectory[:,0], 'b-', linewidth=2)
plt.xlabel("Time (s)"); plt.ylabel("Pole Angle (rad)")
plt.title("Pole Angle Trajectory")
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(time, trajectory[:,1], 'b-', linewidth=2)
plt.xlabel("Time (s)"); plt.ylabel("Angular Velocity (rad/s)")
plt.title("Angular Velocity Trajectory")
plt.grid(True)
plt.tight_layout()
plt.show()
