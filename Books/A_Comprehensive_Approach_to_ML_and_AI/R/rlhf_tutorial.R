# RLHF Tutorial in R
#
# In this simplified RLHF example, the environment is one-dimensional with states 0 to 10.
# The human reward is given by r_human(s) = s. The agent learns a reward model
# r_model(s) = theta0 + theta1 * s via linear regression on collected (state, reward) pairs.
# The policy is logistic: p(right|s) = sigmoid(phi * s + b), and is updated with REINFORCE.

set.seed(1)

# Define environment parameters
stateMin <- 0
stateMax <- 10
startState <- 0
goalState <- 10

# Human reward function
r_human <- function(s) s

# Initialize reward model parameters: r_model(s) = theta0 + theta1 * s
theta <- c(0, 0)

# Initialize policy parameters (scalar)
phi <- 0
b <- 0

# Hyperparameters
numEpisodes <- 1000
maxSteps <- 20
alpha_policy <- 0.01
alpha_model <- 0.001
gamma <- 0.99

# Storage for reward model training
model_states <- c()
model_rewards <- c()

# Sigmoid function
sigmoid <- function(x) 1 / (1 + exp(-x))

# Training Loop (RLHF)
episodeRewards <- numeric(numEpisodes)
for (ep in 1:numEpisodes) {
  s <- startState
  states_ep <- c()
  actions_ep <- c()
  rewards_ep <- c()
  
  for (t in 1:maxSteps) {
    states_ep <- c(states_ep, s)
    p_right <- sigmoid(phi * s + b)
    a <- ifelse(runif(1) < p_right, 1, -1)  # move right (+1) or left (-1)
    actions_ep <- c(actions_ep, a)
    s_next <- min(max(s + a, stateMin), stateMax)
    r <- r_human(s)
    rewards_ep <- c(rewards_ep, r)
    s <- s_next
    if (s == goalState) break
  }
  
  episodeRewards[ep] <- sum(rewards_ep)
  # Compute discounted returns
  T_ep <- length(rewards_ep)
  G <- numeric(T_ep)
  G[T_ep] <- rewards_ep[T_ep]
  if (T_ep > 1) {
    for (t in (T_ep-1):1) {
      G[t] <- rewards_ep[t] + gamma * G[t+1]
    }
  }
  
  # Append data for reward model training
  model_states <- c(model_states, states_ep)
  model_rewards <- c(model_rewards, sapply(states_ep, r_human))
  
  # Update reward model via one gradient descent step (MSE)
  pred_model <- theta[1] + theta[2] * model_states
  error_model <- pred_model - model_rewards
  grad_theta <- c(mean(error_model), mean(error_model * model_states))
  theta <- theta - alpha_model * grad_theta
  
  # Update policy using REINFORCE (using the learned reward model as reward)
  baseline <- mean(G)
  for (t in 1:T_ep) {
    s_t <- states_ep[t]
    p_right <- sigmoid(phi * s_t + b)
    grad_log <- if (actions_ep[t] == 1) (1 - p_right) else (-p_right)
    advantage <- G[t] - baseline
    phi <- phi + alpha_policy * grad_log * s_t * advantage
    b <- b + alpha_policy * grad_log * advantage
  }
  
  if (ep %% 100 == 0) {
    cat(sprintf("Episode %d, Total Reward: %.2f\n", ep, sum(rewards_ep)))
  }
}

cat("Training complete.\n")
cat("Learned reward model parameters (theta):", theta, "\n")
cat("Learned policy parameters (phi, b):", phi, b, "\n")

# Evaluation: Simulate an episode using the greedy policy (move right if p>=0.5)
s <- startState
trajectory <- s
for (t in 1:maxSteps) {
  p_right <- sigmoid(phi * s + b)
  a <- ifelse(p_right >= 0.5, 1, -1)
  s <- min(max(s + a, stateMin), stateMax)
  trajectory <- c(trajectory, s)
  if (s == goalState) break
}
cat("Trajectory (states):", trajectory, "\n")

# Plot the trajectory
plot(trajectory, type = "o", xlab = "Step", ylab = "State", main = "Learned Trajectory")
grid()
