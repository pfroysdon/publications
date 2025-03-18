import numpy as np
import matplotlib.pyplot as plt

class InvertedPendulum:
    """
    A self-contained inverted pendulum simulation.

    The state is defined as:
        state = [theta, theta_dot]
    where theta is the angle (in radians) measured from the upright (theta=0)
    and theta_dot is the angular velocity.

    The observation returned is:
        observation = [cos(theta), sin(theta), theta_dot]

    Dynamics (Euler integration):
        theta_ddot = - (g/l)*sin(theta) + u/(m*l^2)

    Reward:
        reward = 10 + damping_bonus - (theta^2 + 0.5*theta_dot^2 + 0.001*u^2)
    where damping_bonus = 5 if |theta_dot| < 0.1, else 0.

    Termination:
        The episode ends if |theta| > pi/2 (pendulum has fallen) or if
        the maximum number of steps is reached.
    """
    def __init__(self):
        # Physical parameters
        self.m = 1              # mass (kg)
        self.l = 1              # length (m)
        self.g = 9.81           # gravitational acceleration (m/s^2)
        self.dt = 0.02          # time step (s)
        self.maxTorque = 2      # maximum control torque (N*m)
        self.maxEpisodeSteps = 200  # maximum steps per episode
        
        # Internal state: [theta, theta_dot]
        self.state = np.array([0.0, 0.0])
        
        # Step counter
        self.stepCount = 0
        
        # Initialize the pendulum state
        self.reset()

    def reset(self):
        """
        Reinitialize the pendulum state.

        Returns:
            obs: the initial observation [cos(theta), sin(theta), theta_dot]
        """
        # Set theta to a small deviation from upright (uniformly in [-0.1, 0.1] radians)
        theta = np.random.uniform(-0.1, 0.1)
        theta_dot = 0.0
        self.state = np.array([theta, theta_dot])
        self.stepCount = 0
        obs = np.array([np.cos(theta), np.sin(theta), theta_dot])
        return obs

    def step(self, action):
        """
        Apply a control action and update the pendulum state.

        Args:
            action: control torque (scalar)

        Returns:
            obs: next observation [cos(theta), sin(theta), theta_dot]
            reward: reward for this step
            done: boolean flag indicating if the episode is terminated
        """
        # Clip the action to the allowable range.
        u = np.clip(action, -self.maxTorque, self.maxTorque)
        
        # Extract current state.
        theta, theta_dot = self.state
        
        # Compute angular acceleration.
        theta_ddot = - (self.g / self.l) * np.sin(theta) + u / (self.m * self.l**2)
        
        # Update state using Euler integration.
        theta = theta + theta_dot * self.dt
        theta_dot = theta_dot + theta_ddot * self.dt
        
        # Wrap theta to [-pi, pi] using the complex exponential trick.
        theta = np.angle(np.exp(1j * theta))
        
        # Update internal state and counter.
        self.state = np.array([theta, theta_dot])
        self.stepCount += 1
        
        # Construct observation.
        obs = np.array([np.cos(theta), np.sin(theta), theta_dot])
        
        # Compute reward (offset so that near-upright gives high reward).
        damping_bonus = 5 if abs(theta_dot) < 0.1 else 0
        reward = 10 + damping_bonus - (theta**2 + 0.5 * theta_dot**2 + 0.001 * u**2)
        
        # Determine if episode is done (pendulum falls or max steps reached).
        done = (abs(theta) > (np.pi / 2)) or (self.stepCount >= self.maxEpisodeSteps)
        
        return obs, reward, done

    def render(self):
        """
        Visualize the current state of the pendulum.
        The pendulum is drawn as a line from the pivot (origin) to the mass.
        """
        theta, _ = self.state
        x = self.l * np.sin(theta)
        y = self.l * np.cos(theta)
        
        plt.figure(1)
        plt.clf()
        plt.plot([0, x], [0, y], 'b-', linewidth=2)
        plt.plot(x, y, 'ro', markersize=10, markerfacecolor='r')
        plt.axis('equal')
        plt.xlim([-self.l * 1.2, self.l * 1.2])
        plt.ylim([-self.l * 1.2, self.l * 1.2])
        plt.title(f'Step {self.stepCount}')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.pause(0.001)

# Example usage:
if __name__ == '__main__':
    pendulum = InvertedPendulum()
    obs = pendulum.reset()
    
    done = False
    while not done:
        # For example, apply zero torque.
        obs, reward, done = pendulum.step(0)
        print(f"Obs: {obs}, Reward: {reward}, Done: {done}")
        pendulum.render()
