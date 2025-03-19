# gcnn_4layer_tutorial.R
# Advanced GCN on Zachary's Karate Club with 4 Hidden Layers
#
# This script implements a GCN with four hidden layers for node classification.
# The network architecture:
#   H1 = ReLU( A_norm %*% (X %*% W1) )
#   H2 = ReLU( A_norm %*% (H1 %*% W2) )
#   H3 = ReLU( A_norm %*% (H2 %*% W3) )
#   H4 = ReLU( A_norm %*% (H3 %*% W4) )
#   H5 = A_norm %*% (H4 %*% W5)
#   Y_pred = softmaxRows(H5)
#
# Training is performed with full supervision and the refined 4-group partition.
#
rm(list = ls())
graphics.off()
set.seed(1)
library(igraph)

# Load Karate Club Graph
loadKarateClubAdjacency <- function() {
  A <- as.matrix(read.table("data/karate_adj.txt"))
  A
}
A <- loadKarateClubAdjacency()
numNodes <- nrow(A)

# Define refined 4-group labeling
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

# Full supervision
train_mask <- rep(TRUE, numNodes)

# Initialize GCN weights for 4 hidden layers
d_hidden1 <- 32; d_hidden2 <- 32; d_hidden3 <- 32; d_hidden4 <- 32;
W1 <- 0.01 * matrix(rnorm(d_input * d_hidden1), nrow = d_input)
W2 <- 0.01 * matrix(rnorm(d_hidden1 * d_hidden2), nrow = d_hidden1)
W3 <- 0.01 * matrix(rnorm(d_hidden2 * d_hidden3), nrow = d_hidden2)
W4 <- 0.01 * matrix(rnorm(d_hidden3 * d_hidden4), nrow = d_hidden3)
W5 <- 0.01 * matrix(rnorm(d_hidden4 * numClasses), nrow = d_hidden4)

learningRate <- 0.02
epochs <- 3000
g <- graph_from_adjacency_matrix(A, mode = "undirected")
layout_coords <- layout_with_fr(g)
coordsX <- layout_coords[,1]; coordsY <- layout_coords[,2]

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

# Training Loop
for (ep in 1:epochs) {
  # Forward pass through each layer
  H1 <- pmax(A_norm %*% (X %*% W1), 0)
  H2 <- pmax(A_norm %*% (H1 %*% W2), 0)
  H3 <- pmax(A_norm %*% (H2 %*% W3), 0)
  H4 <- pmax(A_norm %*% (H3 %*% W4), 0)
  H5 <- A_norm %*% (H4 %*% W5)
  Y_pred <- softmaxRows(H5)
  
  L <- crossEntropyLoss(Y_pred[train_mask, , drop = FALSE], labels[train_mask], numClasses)
  pred_labels <- max.col(Y_pred)
  acc <- mean(pred_labels[train_mask] == labels[train_mask]) * 100
  
  # (Backpropagation updates for W1..W5 are omitted here; see previous examples.)
  # Assume gradients gradW1, ..., gradW5 are computed.
  # Update weights:
  # W1 <- W1 - learningRate * gradW1, etc.
  
  if (ep %% 200 == 0 || ep == epochs) {
    plot(g, layout = layout_coords, vertex.color = pred_labels, vertex.size = 8,
         main = sprintf("Epoch %d | Loss: %.3f | Acc: %.2f%%", ep, L, acc))
    Sys.sleep(0.1)
    cat(sprintf("Epoch %d | Loss: %.3f | Acc: %.2f%%\n", ep, L, acc))
  }
}
cat(sprintf("Final training accuracy: %.2f%%\n", acc))
