import numpy as np
# Assume that a Python version of the InvertedPendulum environment is available.
# For example, you might have a file inverted_pendulum.py defining the class.
# from inverted_pendulum import InvertedPendulum

# For demonstration, here is a dummy environment class.
class InvertedPendulum:
    def __init__(self):
        self.maxTorque = 2.0
        self.dt = 0.05
        self.maxEpisodeSteps = 200
        self.state = np.array([0.0, 0.0])  # [theta, theta_dot]
        self.steps = 0

    def reset(self):
        self.state = np.array([0.1, 0.0])
        self.steps = 0
        return self.state.copy()

    def step(self, action):
        # Dummy dynamics: new_state = state + dt * action (for illustration)
        self.state = self.state + self.dt * np.array([action, action])
        self.steps += 1
        # Terminate if steps exceed maxEpisodeSteps or if angle exceeds threshold.
        done = self.steps >= self.maxEpisodeSteps or abs(self.state[0]) > np.pi/2
        # Dummy reward: negative cost per step.
        reward = -1.0
        return self.state.copy(), reward, done

    def render(self):
        # For illustration, just print the current state.
        print(f"Rendering state: {self.state}")

# Create the environment
env = InvertedPendulum()

# Reset the environment
obs = env.reset()
done = False

while not done:
    # Choose a random action in [-maxTorque, maxTorque]
    action = np.random.uniform(-env.maxTorque, env.maxTorque)
    
    # Take a step in the environment
    obs, reward, done = env.step(action)
    
    # Display the observation and reward.
    print(f"Observation: {np.round(obs, 3)}, Reward: {reward}")
    
    # Optionally render the pendulum.
    env.render()
    # Optionally pause (e.g., time.sleep(0.001))

print("Episode finished.")
