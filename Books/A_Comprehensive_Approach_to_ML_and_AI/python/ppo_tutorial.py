import numpy as np
import matplotlib.pyplot as plt
import math
# You may need: from inverted_pendulum import InvertedPendulum
# For demonstration, we use the dummy environment defined earlier.
class InvertedPendulum:
    def __init__(self):
        self.maxTorque = 2.0
        self.dt = 0.05
        self.maxEpisodeSteps = 200
        self.state = np.array([0.1, 0.0, 0.0])  # Example observation: [cos(theta), sin(theta), theta_dot]
        self.steps = 0

    def reset(self):
        self.state = np.array([np.cos(0.1), np.sin(0.1), 0.0])
        self.steps = 0
        return self.state.copy()

    def step(self, action):
        # Dummy dynamics for illustration.
        self.state = self.state + self.dt * np.array([0.0, 0.0, action])
        self.steps += 1
        done = self.steps >= self.maxEpisodeSteps
        reward = 1.0  # Dummy reward
        return self.state.copy(), reward, done

    def render(self):
        print(f"Pendulum state: {self.state}")

env = InvertedPendulum()
np.random.seed(1)

# Hyperparameters
num_episodes = 50
max_episode_steps = env.maxEpisodeSteps
gamma = 0.99
ppo_epsilon = 0.2
actor_lr = 1e-3
critic_lr = 1e-3
num_epochs = 5
mini_batch_size = 8
entropy_coef = 0.01
delta = 1e-5

# Network architecture parameters
obs_dim = 3    # e.g., [cos(theta); sin(theta); theta_dot]
action_dim = 1
hidden_size = 16

# Initialize Actor network (parameters as numpy arrays)
actor = {
    'W1': 0.1 * np.random.randn(hidden_size, obs_dim),
    'b1': np.zeros((hidden_size, 1)),
    'W2': 0.1 * np.random.randn(action_dim, hidden_size),
    'b2': np.zeros((action_dim, 1)),
    'log_std': -0.5 * np.ones((action_dim, 1))
}

# Initialize Critic network
critic = {
    'W1': 0.1 * np.random.randn(hidden_size, obs_dim),
    'b1': np.zeros((hidden_size, 1)),
    'W2': 0.1 * np.random.randn(1, hidden_size),
    'b2': 0.0
}

episode_rewards = np.zeros(num_episodes)

# --- Helper functions ---
def actor_forward(actor, s):
    s = s.reshape(-1,1)
    z1 = actor['W1'] @ s + actor['b1']
    h1 = np.maximum(z1, 0)
    mean_a = actor['W2'] @ h1 + actor['b2']
    std_a = np.exp(actor['log_std'])
    return mean_a, std_a

def gaussian_log_prob(a, mean, std):
    a = a.reshape(-1,1)
    logp = -0.5 * np.log(2*np.pi) - np.log(std) - 0.5 * ((a - mean)/std)**2
    return np.sum(logp)

def critic_forward(critic, s):
    s = s.reshape(-1,1)
    z1 = critic['W1'] @ s + critic['b1']
    h1 = np.maximum(z1, 0)
    V = critic['W2'] @ h1 + critic['b2']
    return V[0,0]

def model_loss_actor(actor, obs, actions, old_logp, adv, ppo_epsilon, entropy_coef):
    N = obs.shape[1]
    losses = []
    for i in range(N):
        x = obs[:, i]
        a = actions[:, i]
        mean_a, std_a = actor_forward(actor, x)
        new_logp = gaussian_log_prob(a, mean_a, std_a)
        ratio = np.exp(new_logp - old_logp[i])
        surr1 = ratio * adv[i]
        surr2 = np.clip(ratio, 1-ppo_epsilon, 1+ppo_epsilon) * adv[i]
        losses.append(-min(surr1, surr2))
    actor_loss = np.mean(losses)
    entropy = np.mean(np.exp(actor['log_std']))
    return actor_loss - entropy_coef * entropy

def model_loss_critic(critic, obs, returns):
    N = obs.shape[1]
    errors = []
    for i in range(N):
        x = obs[:, i]
        V_pred = critic_forward(critic, x)
        errors.append((V_pred - returns[i])**2)
    return np.mean(errors)

def finite_diff_gradient(param, lossFunc, delta):
    grad = np.zeros_like(param)
    it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        original = param[idx]
        param[idx] = original + delta
        loss_plus = lossFunc(param)
        param[idx] = original - delta
        loss_minus = lossFunc(param)
        grad[idx] = (loss_plus - loss_minus) / (2 * delta)
        param[idx] = original
        it.iternext()
    return grad

# The update functions use finite differences.
def update_actor(actor, obs, actions, old_logp, adv, ppo_epsilon, actor_lr, entropy_coef, delta):
    loss = model_loss_actor(actor, obs, actions, old_logp, adv, ppo_epsilon, entropy_coef)
    # For each parameter, compute finite-difference gradient and update.
    for key in ['W1', 'b1', 'W2', 'b2', 'log_std']:
        def lossFunc(p):
            temp = actor.copy()
            temp[key] = p
            return model_loss_actor(temp, obs, actions, old_logp, adv, ppo_epsilon, entropy_coef)
        grad = finite_diff_gradient(actor[key], lossFunc, delta)
        actor[key] = actor[key] - actor_lr * grad
    return actor, loss

def update_critic(critic, obs, returns, critic_lr, delta):
    loss = model_loss_critic(critic, obs, returns)
    for key in ['W1', 'b1', 'W2', 'b2']:
        def lossFunc(p):
            temp = critic.copy()
            temp[key] = p
            return model_loss_critic(temp, obs, returns)
        grad = finite_diff_gradient(critic[key], lossFunc, delta)
        critic[key] = critic[key] - critic_lr * grad
    return critic, loss

# --- Main training loop ---
for ep in range(num_episodes):
    obs_batch = []
    actions_batch = []
    rewards_batch = []
    logprobs_batch = []
    values_batch = []
    
    s = env.reset()  # s is a (3,) vector
    done = False
    while not done:
        mean_a, std_a = actor_forward(actor, s)
        a = mean_a + std_a * np.random.randn(*std_a.shape)
        # Clip action within limits
        a = np.clip(a, -env.maxTorque, env.maxTorque)
        logp = gaussian_log_prob(a, mean_a, std_a)
        V = critic_forward(critic, s)
        obs_batch.append(s)
        actions_batch.append(a)
        logprobs_batch.append(logp)
        values_batch.append(V)
        s, r, done = env.step(a)
        rewards_batch.append(r)
    
    T_ep = len(rewards_batch)
    returns = np.zeros(T_ep)
    advantages = np.zeros(T_ep)
    G = 0
    for t in reversed(range(T_ep)):
        G = rewards_batch[t] + gamma * G
        returns[t] = G
        advantages[t] = G - values_batch[t]
    advantages = (advantages - np.mean(advantages)) / (np.std(advantages)+1e-8)
    episode_rewards[ep] = np.sum(rewards_batch)
    
    obs_batch = np.array(obs_batch).T  # shape (obs_dim, T_ep)
    actions_batch = np.array(actions_batch).T  # shape (action_dim, T_ep)
    
    num_samples = T_ep
    indices = np.arange(num_samples)
    for _ in range(num_epochs):
        np.random.shuffle(indices)
        for i in range(0, num_samples, mini_batch_size):
            batch_idx = indices[i:i+mini_batch_size]
            obs_mb = obs_batch[:, batch_idx]
            actions_mb = actions_batch[:, batch_idx]
            old_logp_mb = np.array(logprobs_batch)[batch_idx]
            adv_mb = advantages[batch_idx]
            returns_mb = returns[batch_idx]
            actor, _ = update_actor(actor, obs_mb, actions_mb, old_logp_mb, adv_mb, ppo_epsilon, actor_lr, entropy_coef, delta)
            critic, _ = update_critic(critic, obs_mb, returns_mb, critic_lr, delta)
    print(f"Episode {ep+1}, Total Reward: {episode_rewards[ep]:.2f}")

# Plot training convergence
plt.figure()
plt.plot(np.arange(num_episodes), episode_rewards, 'b-', linewidth=1)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("PPO Convergence on Inverted Pendulum (Finite Differences)")
plt.grid(True)
plt.plot(np.convolve(episode_rewards, np.ones(5)/5, mode='valid'), 'r-', linewidth=2)
plt.legend(["Episode Reward", "Moving Average"])
plt.show()

env.render()

# --- Final Test Run and Trajectory Plot ---
s = env.reset()
trajectory = []
done = False
while not done:
    # For demonstration, use random actions (replace with trained policy if desired)
    action = np.random.uniform(-env.maxTorque, env.maxTorque)
    s, reward, done = env.step(action)
    trajectory.append(env.state.copy())
trajectory = np.array(trajectory)
dt = env.dt
time = np.arange(trajectory.shape[0]) * dt

plt.figure()
plt.subplot(2,1,1)
plt.plot(time, trajectory[:,0], 'b-', linewidth=2)
plt.xlabel("Time (s)")
plt.ylabel("Pole Angle (rad)")
plt.title("Pole Angle Trajectory")
plt.grid(True)
plt.subplot(2,1,2)
plt.plot(time, trajectory[:,1], 'b-', linewidth=2)
plt.xlabel("Time (s)")
plt.ylabel("Angular Velocity (rad/s)")
plt.title("Angular Velocity Trajectory")
plt.grid(True)
plt.tight_layout()
plt.show()
