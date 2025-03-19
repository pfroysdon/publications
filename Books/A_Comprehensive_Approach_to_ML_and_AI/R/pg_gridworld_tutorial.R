# Policy Gradient (REINFORCE) Tutorial on a Gridworld in R
#
# This tutorial demonstrates the REINFORCE algorithm on a simple 5x5 gridworld.
# The environment has 25 states (0-based: 0 to 24). The agent starts at state 0 and
# the goal is at state 24. Available actions are: 1 = Up, 2 = Right, 3 = Down, 4 = Left.
#
# The policy is parameterized by a 4x25 matrix Θ, with p(a|s) = softmax(Θ[,s]).
# The REINFORCE algorithm is used to update Θ.
#
# Convergence is visualized via episode rewards, and the learned policy is displayed on the grid.

set.seed(1)

# -----------------------------
# Environment Parameters
# -----------------------------
gridRows <- 5
gridCols <- 5
numStates <- gridRows * gridCols  # 25 states
startState <- 1  # 1-based index for state 0
goalState <- numStates  # 1-based index for state 24
numActions <- 4
maxSteps <- 50
gamma <- 0.99

# -----------------------------
# Policy Parameters: Θ is a 4x25 matrix
# -----------------------------
Theta <- matrix(rnorm(numActions * numStates, sd = 0.01), nrow = numActions)

# -----------------------------
# Hyperparameters for REINFORCE
# -----------------------------
alpha <- 0.01
numEpisodes <- 10000
episodeRewards <- numeric(numEpisodes)

# Helper functions
softmax <- function(x) {
  x <- x - max(x)
  exp(x) / sum(exp(x))
}

sampleAction <- function(probs) {
  r <- runif(1)
  cumulative <- cumsum(probs)
  which(cumulative >= r)[1]
}

oneHot <- function(a, numActions) {
  vec <- rep(0, numActions)
  vec[a] <- 1
  vec
}

# Step function for gridworld
step_env <- function(s, action, gridRows, gridCols, goalState) {
  # s: current state (1-based)
  pos <- arrayInd(s, .dim = c(gridRows, gridCols))
  row <- pos[1]; col <- pos[2]
  if (action == 1) {         # up
    newRow <- max(row - 1, 1); newCol <- col
  } else if (action == 2) {    # right
    newRow <- row; newCol <- min(col + 1, gridCols)
  } else if (action == 3) {    # down
    newRow <- min(row + 1, gridRows); newCol <- col
  } else if (action == 4) {    # left
    newRow <- row; newCol <- max(col - 1, 1)
  } else {
    newRow <- row; newCol <- col
  }
  s_next <- (newRow - 1) * gridCols + newCol
  r <- ifelse(s_next == goalState, 10, -1)
  done <- (s_next == goalState)
  list(s_next = s_next, r = r, done = done)
}

# -----------------------------
# Main Training Loop (REINFORCE)
# -----------------------------
for (ep in 1:numEpisodes) {
  s <- startState
  states <- c()
  actions <- c()
  rewards <- c()
  
  for (t in 1:maxSteps) {
    states <- c(states, s)
    probs <- softmax(Theta[, s])
    a <- sampleAction(probs)
    actions <- c(actions, a)
    res <- step_env(s, a, gridRows, gridCols, goalState)
    rewards <- c(rewards, res$r)
    s <- res$s_next
    if (res$done) break
  }
  
  episodeRewards[ep] <- sum(rewards)
  
  # Compute discounted returns
  T_ep <- length(rewards)
  G <- numeric(T_ep)
  G[T_ep] <- rewards[T_ep]
  if (T_ep > 1) {
    for (t in (T_ep-1):1) {
      G[t] <- rewards[t] + gamma * G[t+1]
    }
  }
  
  # Policy update using REINFORCE
  for (t in 1:T_ep) {
    s_t <- states[t]
    a_t <- actions[t]
    grad_log <- oneHot(a_t, numActions) - softmax(Theta[, s_t])
    Theta[, s_t] <- Theta[, s_t] + alpha * grad_log * G[t]
  }
}

# Plot convergence of episode rewards
plot(1:numEpisodes, episodeRewards, type = "l", col = "blue", lwd = 1,
     xlab = "Episode", ylab = "Total Reward", main = "Policy Gradient Convergence")
lines(1:numEpisodes, zoo::rollmean(episodeRewards, 50, fill = NA), col = "red", lwd = 2)
legend("topright", legend = c("Episode Reward", "Moving Average"), col = c("blue", "red"), lty = 1)

# -----------------------------
# Evaluate Learned Policy: Greedy Trajectory
# -----------------------------
s <- startState
optimalPath <- s
for (t in 1:maxSteps) {
  probs <- softmax(Theta[, s])
  a <- which.max(probs)
  res <- step_env(s, a, gridRows, gridCols, goalState)
  optimalPath <- c(optimalPath, res$s_next)
  s <- res$s_next
  if (res$done) break
}
cat("Optimal path (1-based):", optimalPath, "\n")
cat("Optimal path (0-based):", optimalPath - 1, "\n")

# -----------------------------
# Visualize the Grid and Learned Policy
# -----------------------------
library(igraph)
# Build edge list from valid transitions in the reward matrix R (simulate from original MATLAB R)
R_mat <- matrix(c(-1, -1, -1, -1,  0, -1,
                  -1, -1, -1,  0, -1, 100,
                  -1, -1, -1,  0, -1, -1,
                  -1,  0,  0, -1,  0, -1,
                   0, -1, -1,  0, -1, 100,
                  -1,  0, -1, -1,  0, 100), nrow = numStates, byrow = TRUE)
edgesFrom <- c()
edgesTo <- c()
for (i in 1:numStates) {
  for (j in 1:numStates) {
    if (R_mat[i,j] != -1) {
      edgesFrom <- c(edgesFrom, i)
      edgesTo <- c(edgesTo, j)
    }
  }
}
G <- graph_from_data_frame(data.frame(from = edgesFrom, to = edgesTo), directed = TRUE)
plot(G, layout = layout_as_tree(G, root = 1), vertex.label = 0:(numStates-1),
     main = "Gridworld Directed Graph (0-based labels)")
