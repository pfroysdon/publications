# Q-Learning Tutorial in R
#
# This script demonstrates Q-learning on a directed graph with 6 states (0-based: 0 to 5).
# The reward matrix R defines valid transitions and rewards. Q-values are learned via:
#
#   Q(s,a) = Q(s,a) + alpha [ R(s,a) + gamma max_a' Q(s',a') - Q(s,a) ]
#
# Each episode starts at state 2 (0-based) and ends when state 5 is reached.
# After training, the optimal (greedy) policy is extracted and the directed graph is plotted.
# The optimal path is highlighted.

set.seed(1)

# -----------------------------
# Define Reward Matrix R (6x6) using 1-based indexing.
# States (0-based): 0 to 5 correspond to indices 1 to 6.
R <- matrix(c(-1, -1, -1, -1,  0, -1,
              -1, -1, -1,  0, -1, 100,
              -1, -1, -1,  0, -1, -1,
              -1,  0,  0, -1,  0, -1,
               0, -1, -1,  0, -1, 100,
              -1,  0, -1, -1,  0, 100), nrow = 6, byrow = TRUE)

numStates <- 6
startState <- 2  # (0-based state 2, MATLAB index = 3) -> use 0-based here: 2
goalState <- 5   # (0-based)

alpha <- 0.1
gamma <- 0.8
epsilon <- 0.1
numEpisodes <- 400

# Initialize Q-table: 6x6 matrix
Q <- matrix(0, nrow = numStates, ncol = numStates)
Q_out <- numeric(numEpisodes)

for (episode in 1:numEpisodes) {
  s <- startState  # starting state (0-based)
  while (s != goalState) {
    s_idx <- s + 1  # convert to 1-based index for R
    validActions <- which(R[s_idx, ] != -1) - 1  # back to 0-based
    if (length(validActions) == 0) break
    
    # Epsilon-greedy selection
    if (runif(1) < epsilon) {
      a <- sample(validActions, 1)
    } else {
      Q_valid <- Q[s_idx, validActions + 1]
      a <- validActions[which.max(Q_valid)]
    }
    a_idx <- a + 1
    s_next <- a  # next state (0-based)
    rew <- R[s_idx, a_idx]
    
    if (s_next == goalState) {
      Q[s_idx, a_idx] <- Q[s_idx, a_idx] + alpha * (rew - Q[s_idx, a_idx])
      break
    } else {
      next_idx <- s_next + 1
      Q[s_idx, a_idx] <- Q[s_idx, a_idx] + alpha * (rew + gamma * max(Q[next_idx, ]) - Q[s_idx, a_idx])
    }
    s <- s_next
  }
  Q_out[episode] <- norm(Q, type = "2")
}

cat("Learned Q-Table:\n")
print(Q)

plot(Q_out, type = "l", col = "blue", lwd = 2, xlab = "Episode", ylab = "||Q||_2",
     main = "Convergence of Q-Values")

# Extract the Optimal Path from state 2 to state 5
optimalPath <- c(startState)
s <- startState
while (s != goalState) {
  s_idx <- s + 1
  a <- which.max(Q[s_idx, ]) - 1  # convert back to 0-based
  optimalPath <- c(optimalPath, a)
  s <- a
}
cat("Optimal path (0-based):", optimalPath, "\n")

# Plot the directed graph using igraph
library(igraph)
edgesFrom0 <- c(); edgesTo0 <- c(); edgeWeights <- c()
for (i in 1:numStates) {
  for (j in 1:numStates) {
    if (R[i, j] != -1) {
      edgesFrom0 <- c(edgesFrom0, i - 1)  # 0-based
      edgesTo0 <- c(edgesTo0, j - 1)
      edgeWeights <- c(edgeWeights, R[i, j])
    }
  }
}
# Create graph (convert to 1-based for igraph)
G <- graph_from_data_frame(data.frame(from = edgesFrom0 + 1, to = edgesTo0 + 1, weight = edgeWeights), directed = TRUE)
plot(G, layout = layout_as_tree(G, root = startState + 1),
     vertex.label = 0:(numStates - 1), edge.color = "blue", main = "Directed Graph with Optimal Path")

# Highlight optimal path edges in red
optimalPathIndices <- optimalPath + 1  # convert to 1-based
for (k in 1:(length(optimalPathIndices) - 1)) {
  s_from <- optimalPathIndices[k]
  s_to <- optimalPathIndices[k + 1]
  E(G)[from(s_from) & to(s_to)]$color <- "red"
}
plot(G, layout = layout_as_tree(G, root = startState + 1),
     vertex.label = 0:(numStates - 1), edge.width = 2,
     main = "Directed Graph with Optimal Path (red edges)")
