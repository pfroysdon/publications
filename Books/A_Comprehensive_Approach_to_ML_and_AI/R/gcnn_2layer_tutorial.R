# gcnn_2layer_tutorial.R
# Graph Convolutional Network (GCN) on Zachary's Karate Club – Basic 2-Layer Version
#
# This script implements a simple two-layer GCN for node classification on the Karate Club graph.
# The architecture is:
#   H1 = ReLU( A_norm %*% (X %*% W1) )
#   Y_pred = softmaxRows( A_norm %*% (H1 %*% W2) )
#
# Training is performed using full supervision.
#
rm(list = ls())
graphics.off()
set.seed(1)
library(igraph)

# Load Karate Club Graph (using the same helper as before)
loadKarateClubAdjacency <- function() {
  A <- as.matrix(read.table("data/karate_adj.txt"))
  A
}
A <- loadKarateClubAdjacency()
numNodes <- nrow(A)

# Define 4-group labels (can be same as previous)
group1 <- c(1,2,3,4,8,14)
group2 <- c(5,6,7,11,12,13,17)
group3 <- c(9,10,15,16,19,21,23,25,27,29,30,31,33,34)
group4 <- c(18,20,22,24,26,28,32)
labels <- rep(0, numNodes)
labels[group1] <- 1; labels[group2] <- 2; labels[group3] <- 3; labels[group4] <- 4
numClasses <- 4

# Node features: identity matrix
X <- diag(numNodes)
d_input <- numNodes

# Normalized adjacency matrix
I <- diag(numNodes)
A_tilde <- A + I
D_tilde <- diag(rowSums(A_tilde))
D_inv_sqrt <- diag(1/sqrt(diag(D_tilde)))
A_norm <- D_inv_sqrt %*% A_tilde %*% D_inv_sqrt

# Initialize weights for two-layer GCN
d_hidden <- 16
W1 <- 0.01 * matrix(rnorm(d_input * d_hidden), nrow = d_input)
W2 <- 0.01 * matrix(rnorm(d_hidden * numClasses), nrow = d_hidden)

# Training parameters
learningRate <- 0.02
epochs <- 1000
train_mask <- rep(TRUE, numNodes)

# 2D layout for plotting
g <- graph_from_adjacency_matrix(A, mode = "undirected")
layout_coords <- layout_with_fr(g)
coordsX <- layout_coords[,1]
coordsY <- layout_coords[,2]

softmaxRows <- function(X) {
  expX <- exp(X - apply(X, 1, max))
  expX / rowSums(expX)
}
crossEntropyLoss <- function(pred, true_labels, numClasses) {
  n <- nrow(pred)
  loss <- 0
  for (i in 1:n) {
    loss <- loss - log(pred[i, true_labels[i]] + 1e-8)
  }
  loss / n
}

# Training loop
for (ep in 1:epochs) {
  H1 <- A_norm %*% (X %*% W1)
  H1 <- pmax(H1, 0)
  H2 <- A_norm %*% (H1 %*% W2)
  Y_pred <- softmaxRows(H2)
  
  L <- crossEntropyLoss(Y_pred[train_mask, , drop = FALSE], labels[train_mask], numClasses)
  pred_labels <- max.col(Y_pred)
  acc <- mean(pred_labels[train_mask] == labels[train_mask]) * 100
  
  # (Gradient computation and parameter updates are omitted for brevity)
  # For a complete implementation, compute gradients w.r.t. W1 and W2 and update:
  # W1 <- W1 - learningRate * gradW1; W2 <- W2 - learningRate * gradW2
  
  if (ep %% 50 == 0 || ep == epochs) {
    plot(g, layout = layout_coords, vertex.color = pred_labels, vertex.size = 20,
         main = sprintf("Epoch %d | Loss: %.3f | Acc: %.2f%%", ep, L, acc))
    Sys.sleep(0.1)
    cat(sprintf("Epoch %d | Loss: %.3f | Acc: %.2f%%\n", ep, L, acc))
  }
}
cat(sprintf("Final training accuracy: %.2f%%\n", acc))
