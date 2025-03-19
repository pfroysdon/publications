# ddpg_pendulum_tutorial.R
# This script implements a simplified DDPG algorithm for an inverted pendulum.
# It uses finite-difference gradient estimation (for educational purposes).
# Note: This implementation is simplified and may run slowly.

rm(list=ls())
graphics.off()
set.seed(1)
library(zoo)  # for movmean

# Define an Inverted Pendulum environment in R
InvertedPendulum <- function() {
  env <- list()
  env$m <- 1
  env$l <- 1
  env$g <- 9.81
  env$dt <- 0.02
  env$maxTorque <- 2
  env$maxEpisodeSteps <- 200
  env$state <- c(0, 0)
  env$stepCount <- 0
  env$reset <- function() {
    theta <- runif(1, -0.1, 0.1)
    theta_dot <- 0
    env$state <<- c(theta, theta_dot)
    env$stepCount <<- 0
    c(cos(theta), sin(theta), theta_dot)
  }
  env$step <- function(action) {
    a <- max(min(action, env$maxTorque), -env$maxTorque)
    theta <- env$state[1]
    theta_dot <- env$state[2]
    theta_ddot <- - (env$g / env$l) * sin(theta) + a / (env$m * env$l^2)
    theta <- theta + theta_dot * env$dt
    theta_dot <- theta_dot + theta_ddot * env$dt
    theta <- atan2(sin(theta), cos(theta))
    env$state <<- c(theta, theta_dot)
    env$stepCount <<- env$stepCount + 1
    obs <- c(cos(theta), sin(theta), theta_dot)
    damping_bonus <- ifelse(abs(theta_dot) < 0.1, 5, 0)
    reward <- 10 + damping_bonus - (theta^2 + 0.5 * theta_dot^2 + 0.001 * a^2)
    done <- (abs(theta) > (pi/2)) || (env$stepCount >= env$maxEpisodeSteps)
    list(obs = obs, reward = reward, done = done)
  }
  env
}

env <- InvertedPendulum()

# Hyperparameters
numEpisodes <- 20
maxSteps <- env$maxEpisodeSteps
gamma <- 0.99
actorLR <- 0.001
criticLR <- 0.001
tau <- 0.01
batchSize <- 8
replayCapacity <- 1000
explorationNoise <- 0.01
delta <- 1e-5

obs_dim <- 3
action_dim <- 1
hidden_size <- 16

# Initialize Actor network
actor <- list()
actor$W1 <- 0.1 * matrix(rnorm(hidden_size * obs_dim), nrow = hidden_size)
actor$b1 <- matrix(0, nrow = hidden_size, ncol = 1)
actor$W2 <- 0.1 * matrix(rnorm(action_dim * hidden_size), nrow = action_dim)
actor$b2 <- matrix(0, nrow = action_dim, ncol = 1)

# Initialize Critic network
critic <- list()
critic$W1 <- 0.1 * matrix(rnorm(hidden_size * (obs_dim + action_dim)), nrow = hidden_size)
critic$b1 <- matrix(0, nrow = hidden_size, ncol = 1)
critic$W2 <- 0.1 * matrix(rnorm(1 * hidden_size), nrow = 1)
critic$b2 <- 0

# Initialize Target networks as copies
actor_target <- actor
critic_target <- critic

# Initialize Replay Buffer
buffer <- list(count = 0,
               capacity = replayCapacity,
               states = matrix(0, nrow = obs_dim, ncol = replayCapacity),
               actions = matrix(0, nrow = action_dim, ncol = replayCapacity),
               rewards = rep(0, replayCapacity),
               next_states = matrix(0, nrow = obs_dim, ncol = replayCapacity),
               dones = rep(0, replayCapacity))

# Activation functions
sigmoid <- function(x) { 1 / (1 + exp(-x)) }
relu <- function(x) { pmax(0, x) }

actor_forward <- function(actor, s) {
  z1 <- actor$W1 %*% s + actor$b1
  h1 <- relu(z1)
  actor$W2 %*% h1 + actor$b2
}

critic_forward <- function(critic, sa) {
  z1 <- critic$W1 %*% sa + critic$b1
  h1 <- relu(z1)
  critic$W2 %*% h1 + critic$b2
}

finiteDiffGradient <- function(param, lossFunc, delta) {
  g <- array(0, dim = dim(param))
  for (i in 1:length(param)) {
    orig <- param[i]
    param[i] <- orig + delta
    loss_plus <- lossFunc(param)
    param[i] <- orig - delta
    loss_minus <- lossFunc(param)
    g[i] <- (loss_plus - loss_minus) / (2 * delta)
    param[i] <- orig
  }
  g
}

update_actor_loss_wrapper <- function(actor, states, critic) {
  N <- ncol(states)
  loss <- 0
  for (i in 1:N) {
    s <- matrix(states[, i], ncol = 1)
    a <- actor_forward(actor, s)
    Q_val <- critic_forward(critic, rbind(s, a))
    loss <- loss - Q_val
  }
  loss / N
}

update_actor <- function(actor, states, critic, actorLR, delta) {
  actor$W1 <- actor$W1 - actorLR * finiteDiffGradient(actor$W1, function(W) update_actor_loss_wrapper(modifyList(actor, list(W1 = W)), states, critic), delta)
  actor$b1 <- actor$b1 - actorLR * finiteDiffGradient(actor$b1, function(b) update_actor_loss_wrapper(modifyList(actor, list(b1 = b)), states, critic), delta)
  actor$W2 <- actor$W2 - actorLR * finiteDiffGradient(actor$W2, function(W) update_actor_loss_wrapper(modifyList(actor, list(W2 = W)), states, critic), delta)
  actor$b2 <- actor$b2 - actorLR * finiteDiffGradient(actor$b2, function(b) update_actor_loss_wrapper(modifyList(actor, list(b2 = b)), states, critic), delta)
  list(actor = actor)
}

update_critic_loss_wrapper <- function(critic, states, actions, target_Q) {
  N <- ncol(states)
  loss <- 0
  for (i in 1:N) {
    s <- matrix(states[, i], ncol = 1)
    a <- matrix(actions[, i], ncol = 1)
    Q_val <- critic_forward(critic, rbind(s, a))
    loss <- loss + (Q_val - target_Q[i])^2
  }
  loss / N
}

update_critic <- function(critic, states, actions, target_Q, criticLR, delta) {
  critic$W1 <- critic$W1 - criticLR * finiteDiffGradient(critic$W1, function(W) update_critic_loss_wrapper(modifyList(critic, list(W1 = W)), states, actions, target_Q), delta)
  critic$b1 <- critic$b1 - criticLR * finiteDiffGradient(critic$b1, function(b) update_critic_loss_wrapper(modifyList(critic, list(b1 = b)), states, actions, target_Q), delta)
  critic$W2 <- critic$W2 - criticLR * finiteDiffGradient(critic$W2, function(W) update_critic_loss_wrapper(modifyList(critic, list(W2 = W)), states, actions, target_Q), delta)
  critic$b2 <- critic$b2 - criticLR * finiteDiffGradient(critic$b2, function(b) update_critic_loss_wrapper(modifyList(critic, list(b2 = b)), states, actions, target_Q), delta)
  list(critic = critic)
}

soft_update <- function(net, target_net, tau) {
  for (fld in names(net)) {
    target_net[[fld]] <- tau * net[[fld]] + (1 - tau) * target_net[[fld]]
  }
  target_net
}

# Main Training Loop
episodeRewards <- numeric(numEpisodes)
for (ep in 1:numEpisodes) {
  s <- env$reset()
  totalReward <- 0
  for (t in 1:maxSteps) {
    s_vec <- matrix(s, ncol = 1)
    a <- actor_forward(actor, s_vec)
    a <- a + explorationNoise * rnorm(length(a))
    a <- max(min(a, env$maxTorque), -env$maxTorque)
    stepResult <- env$step(a)
    s_next <- stepResult$obs
    r <- stepResult$reward
    done <- stepResult$done
    totalReward <- totalReward + r
    buffer$count <- buffer$count + 1
    idx <- ((buffer$count - 1) %% replayCapacity) + 1
    buffer$states[, idx] <- s
    buffer$actions[, idx] <- a
    buffer$rewards[idx] <- r
    buffer$next_states[, idx] <- s_next
    buffer$dones[idx] <- as.numeric(done)
    s <- s_next
    if (done) break
    if (buffer$count >= batchSize) {
      indices <- sample(1:min(buffer$count, replayCapacity), batchSize, replace = TRUE)
      batch <- list(
        states = buffer$states[, indices, drop = FALSE],
        actions = buffer$actions[, indices, drop = FALSE],
        rewards = buffer$rewards[indices],
        next_states = buffer$next_states[, indices, drop = FALSE],
        dones = buffer$dones[indices]
      )
      target_Q <- numeric(batchSize)
      for (i in 1:batchSize) {
        s_next_i <- matrix(batch$next_states[, i], ncol = 1)
        r_i <- batch$rewards[i]
        done_i <- batch$dones[i]
        a_next <- actor_forward(actor_target, s_next_i)
        Q_next <- critic_forward(critic_target, rbind(s_next_i, a_next))
        if (done_i == 1) {
          target_Q[i] <- r_i
        } else {
          target_Q[i] <- r_i + gamma * Q_next
        }
      }
      critic_update <- update_critic(critic, batch$states, batch$actions, target_Q, criticLR, delta)
      critic <- critic_update$critic
      actor_update <- update_actor(actor, batch$states, critic, actorLR, delta)
      actor <- actor_update$actor
      actor_target <- soft_update(actor, actor_target, tau)
      critic_target <- soft_update(critic, critic_target, tau)
    }
  }
  episodeRewards[ep] <- totalReward
  cat(sprintf("Episode %d, Total Reward: %.2f\n", ep, totalReward))
}

# Plot convergence of episode rewards
plot(1:numEpisodes, episodeRewards, type = "l", lwd = 1, xlab = "Episode", ylab = "Total Reward",
     main = "DDPG Convergence on Inverted Pendulum")
lines(1:numEpisodes, rollmean(episodeRewards, 10, fill = NA), col = "red", lwd = 2)
legend("bottomright", legend = c("Episode Reward", "Moving Average"), col = c("blue", "red"), lwd = c(1, 2))

# Final test run and trajectory plot
s <- env$reset()
trajectory <- matrix(nrow = 0, ncol = 2)
done <- FALSE
while (!done) {
  s_vec <- matrix(s, ncol = 1)
  a <- actor_forward(actor, s_vec)
  a <- max(min(a, env$maxTorque), -env$maxTorque)
  stepResult <- env$step(a)
  s <- stepResult$obs
  done <- stepResult$done
  trajectory <- rbind(trajectory, env$state)
}

dt <- env$dt
time <- seq(0, by = dt, length.out = nrow(trajectory))
par(mfrow = c(2, 1))
plot(time, trajectory[, 1], type = "l", col = "blue", lwd = 2,
     xlab = "Time (s)", ylab = "Pole Angle (rad)", main = "Pole Angle Trajectory")
plot(time, trajectory[, 2], type = "l", col = "blue", lwd = 2,
     xlab = "Time (s)", ylab = "Angular Velocity (rad/s)", main = "Angular Velocity Trajectory")
