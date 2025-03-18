#!/usr/bin/env python3
"""
ddpg_tutorial_pendulum.py
-------------------------
This tutorial implements DDPG for an inverted pendulum using finite-difference gradients.
A custom InvertedPendulum environment is defined.
The actor network maps observations to a control torque,
and the critic network estimates Q-values.
Finite-difference gradient estimation is used for parameter updates.
"""

import numpy as np
import matplotlib.pyplot as plt

# Inverted Pendulum Environment
class InvertedPendulum:
    def __init__(self):
        self.dt = 0.02
        self.maxEpisodeSteps = 200
        self.maxTorque = 2.0
        self.reset()
    
    def reset(self):
        # State: [theta, theta_dot] (we return observation: [cos(theta), sin(theta), theta_dot])
        self.theta = np.random.uniform(-0.1, 0.1)
        self.theta_dot = 0.0
        self.steps = 0
        return self._get_obs()
    
    def step(self, torque):
        self.steps += 1
        # Simple pendulum dynamics
        g = 9.8; l = 1.0; m = 1.0
        theta_acc = (g / l) * np.sin(self.theta) + torque / (m * l**2)
        self.theta_dot += theta_acc * self.dt
        self.theta += self.theta_dot * self.dt
        # Reward: higher when theta is near 0 (upright)
        reward = -abs(self.theta)
        done = self.steps >= self.maxEpisodeSteps or abs(self.theta) > np.pi/2
        return self._get_obs(), reward, done
    
    def _get_obs(self):
        return np.array([np.cos(self.theta), np.sin(self.theta), self.theta_dot])

# Activation functions
def relu(z):
    return np.maximum(0, z)

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

# Actor network forward pass
def actor_forward(actor, s):
    z1 = actor['W1'] @ s + actor['b1']
    h1 = relu(z1)
    a = actor['W2'] @ h1 + actor['b2']
    return a

# Critic network forward pass
def critic_forward(critic, sa):
    z1 = critic['W1'] @ sa + critic['b1']
    h1 = relu(z1)
    Q = critic['W2'] @ h1 + critic['b2']
    return Q

# Soft update function
def soft_update(net, target_net, tau):
    for key in net:
        target_net[key] = tau * net[key] + (1 - tau) * target_net[key]
    return target_net

# Dummy wrappers for actor and critic updates using finite differences
def update_actor(actor, states, critic, actor_lr, delta):
    N = states.shape[1]
    loss = 0
    for i in range(N):
        s = states[:, i]
        a = actor_forward(actor, s)
        Q_val = critic_forward(critic, np.concatenate([s, a]))
        loss -= Q_val
    loss /= N
    # Finite-difference update for each parameter
    actor['W1'] -= actor_lr * finite_diff_gradient(actor['W1'], lambda p: actor_loss_wrapper({**actor, 'W1': p}, states, critic), delta)
    actor['b1'] -= actor_lr * finite_diff_gradient(actor['b1'], lambda p: actor_loss_wrapper({**actor, 'b1': p}, states, critic), delta)
    actor['W2'] -= actor_lr * finite_diff_gradient(actor['W2'], lambda p: actor_loss_wrapper({**actor, 'W2': p}, states, critic), delta)
    actor['b2'] -= actor_lr * finite_diff_gradient(actor['b2'], lambda p: actor_loss_wrapper({**actor, 'b2': p}, states, critic), delta)
    return actor, loss

def actor_loss_wrapper(actor, states, critic):
    N = states.shape[1]
    loss = 0
    for i in range(N):
        s = states[:, i]
        a = actor_forward(actor, s)
        Q_val = critic_forward(critic, np.concatenate([s, a]))
        loss -= Q_val
    return loss / N

def update_critic(critic, states, actions, target_Q, critic_lr, delta):
    N = states.shape[1]
    loss = 0
    for i in range(N):
        s = states[:, i]
        a = actions[:, i]
        Q_val = critic_forward(critic, np.concatenate([s, a]))
        loss += (Q_val - target_Q[i])**2
    loss /= N
    critic['W1'] -= critic_lr * finite_diff_gradient(critic['W1'], lambda p: critic_loss_wrapper({**critic, 'W1': p}, states, actions, target_Q), delta)
    critic['b1'] -= critic_lr * finite_diff_gradient(critic['b1'], lambda p: critic_loss_wrapper({**critic, 'b1': p}, states, actions, target_Q), delta)
    critic['W2'] -= critic_lr * finite_diff_gradient(critic['W2'], lambda p: critic_loss_wrapper({**critic, 'W2': p}, states, actions, target_Q), delta)
    critic['b2'] -= critic_lr * finite_diff_gradient(critic['b2'], lambda p: critic_loss_wrapper({**critic, 'b2': p}, states, actions, target_Q), delta)
    return critic, loss

def critic_loss_wrapper(critic, states, actions, target_Q):
    N = states.shape[1]
    loss = 0
    for i in range(N):
        s = states[:, i]
        a = actions[:, i]
        Q_val = critic_forward(critic, np.concatenate([s, a]))
        loss += (Q_val - target_Q[i])**2
    return loss / N

# Main DDPG training loop for inverted pendulum using finite differences
env = InvertedPendulum()

numEpisodes = 20
maxSteps = env.maxEpisodeSteps
gamma = 0.99
actorLR = 0.001
criticLR = 0.001
tau = 0.01
batchSize = 8
replayCapacity = 1000
explorationNoise = 0.01
delta = 1e-5

obs_dim = 3
action_dim = 1
hidden_size = 16

# Initialize networks
actor = {
    'W1': 0.1 * np.random.randn(hidden_size, obs_dim),
    'b1': np.zeros((hidden_size, 1)),
    'W2': 0.1 * np.random.randn(action_dim, hidden_size),
    'b2': np.zeros((action_dim, 1))
}
critic = {
    'W1': 0.1 * np.random.randn(hidden_size, obs_dim + action_dim),
    'b1': np.zeros((hidden_size, 1)),
    'W2': 0.1 * np.random.randn(1, hidden_size),
    'b2': 0
}
actor_target = actor.copy()
critic_target = critic.copy()

# Initialize replay buffer
buffer = {'count': 0, 'states': np.zeros((obs_dim, replayCapacity)),
          'actions': np.zeros((action_dim, replayCapacity)),
          'rewards': np.zeros(replayCapacity),
          'next_states': np.zeros((obs_dim, replayCapacity)),
          'dones': np.zeros(replayCapacity)}
episodeRewards = []

for ep in range(numEpisodes):
    s = env.reset()
    totalReward = 0
    for t in range(maxSteps):
        a = actor_forward(actor, s.reshape(-1,1))
        a = a + explorationNoise * np.random.randn(*a.shape)
        a = np.clip(a, -env.maxTorque, env.maxTorque)
        s_next, r, done = env.step(a)
        totalReward += r
        buffer['count'] += 1
        idx = (buffer['count'] - 1) % replayCapacity
        buffer['states'][:, idx] = s
        buffer['actions'][:, idx] = a.flatten()
        buffer['rewards'][idx] = r
        buffer['next_states'][:, idx] = s_next
        buffer['dones'][idx] = done
        s = s_next
        if done:
            break
        if buffer['count'] >= batchSize:
            indices = np.random.choice(min(buffer['count'], replayCapacity), batchSize, replace=False)
            batch_states = buffer['states'][:, indices]
            batch_actions = buffer['actions'][:, indices]
            batch_rewards = buffer['rewards'][indices]
            batch_next_states = buffer['next_states'][:, indices]
            batch_dones = buffer['dones'][indices]
            target_Q = np.zeros(batchSize)
            for i in range(batchSize):
                s_next_i = batch_next_states[:, i]
                r_i = batch_rewards[i]
                done_i = batch_dones[i]
                a_next = actor_forward(actor_target, s_next_i.reshape(-1,1))
                Q_next = critic_forward(critic_target, np.concatenate([s_next_i, a_next.flatten()]))
                target_Q[i] = r_i if done_i else r_i + gamma * Q_next
            critic, critic_loss = update_critic(critic, batch_states, batch_actions, target_Q, criticLR, delta)
            actor, actor_loss = update_actor(actor, batch_states, critic, actorLR, delta)
            actor_target = soft_update(actor, actor_target, tau)
            critic_target = soft_update(critic, critic_target, tau)
    episodeRewards.append(totalReward)
    print(f"Episode {ep+1}, Total Reward: {totalReward:.2f}")

plt.figure()
plt.plot(np.arange(1, numEpisodes+1), episodeRewards, 'b-', linewidth=1)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("DDPG Convergence on Inverted Pendulum")
plt.grid(True)
plt.plot(np.arange(1, numEpisodes+1), np.convolve(episodeRewards, np.ones(10)/10, mode='same'), 'r-', linewidth=2, label="Moving Average")
plt.legend()
plt.show()

# Final test run: record trajectory
s = env.reset()
trajectory = []
done = False
while not done:
    a = actor_forward(actor, s.reshape(-1,1))
    a = np.clip(a, -env.maxTorque, env.maxTorque)
    s, r, done = env.step(a)
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
