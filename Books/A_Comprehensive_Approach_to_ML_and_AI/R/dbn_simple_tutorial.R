# dbn_simple_tutorial.R
# This script demonstrates a simple DBN using one RBM layer on toy data.
# It trains an RBM with Contrastive Divergence (CD-1) and visualizes the hidden activations and weight matrix.

rm(list=ls())
graphics.off()
set.seed(1)

# Parameters
numEpochs <- 100
learningRate <- 0.05
numHidden <- 10

# Generate toy data: 200 samples with 20 features (normalized)
X <- matrix(runif(200 * 20), nrow = 200, ncol = 20)

# Sigmoid activation function
sigmoid <- function(x) { 1 / (1 + exp(-x)) }

# Train RBM using CD-1
trainRBM <- function(X, numHidden, numEpochs, learningRate) {
  n_samples <- nrow(X)
  n_features <- ncol(X)
  W <- 0.1 * matrix(rnorm(n_features * numHidden), nrow = n_features)
  b_visible <- rep(0, n_features)
  b_hidden <- rep(0, numHidden)
  k <- 1  # number of Gibbs steps (CD-1)
  
  for (epoch in 1:numEpochs) {
    for (i in 1:n_samples) {
      v0 <- X[i, ]
      h0_prob <- sigmoid(v0 %*% W + b_hidden)
      h0 <- as.numeric(h0_prob > runif(numHidden))
      vk <- v0
      hk <- h0
      for (step in 1:k) {
        vk_prob <- sigmoid(hk %*% t(W) + b_visible)
        vk <- as.numeric(vk_prob > runif(n_features))
        hk_prob <- sigmoid(vk %*% W + b_hidden)
        hk <- as.numeric(hk_prob > runif(numHidden))
      }
      dW <- (outer(v0, h0_prob)) - (outer(vk, hk_prob))
      db_visible <- v0 - vk
      db_hidden <- h0_prob - hk_prob
      W <- W + learningRate * dW
      b_visible <- b_visible + learningRate * db_visible
      b_hidden <- b_hidden + learningRate * db_hidden
    }
    if (epoch %% 10 == 0) {
      cat(sprintf("Epoch %d complete.\n", epoch))
    }
  }
  list(W = W, b_hidden = b_hidden, b_visible = b_visible)
}

rbm <- trainRBM(X, numHidden, numEpochs, learningRate)

# Obtain hidden representations (activations)
H <- sigmoid(X %*% rbm$W + matrix(rep(rbm$b_hidden, nrow(X)), nrow = nrow(X), byrow = TRUE))

# Visualize hidden activations using a heatmap (first 50 samples)
heatmap(H[1:50, ], Rowv = NA, Colv = NA, scale = "none",
        main = "Hidden Activations from the RBM Layer")

# Display the learned weight matrix as an image
image(rbm$W, col = gray.colors(256), main = "Learned Weight Matrix",
      xlab = "Hidden Units", ylab = "Input Features")
