# dbn_xor_tutorial.R
# This script demonstrates a DBN for XOR-like classification.
# It generates an XOR-like dataset, pretrains an RBM, trains a logistic regression classifier
# on the RBM features, fine-tunes the DBN with supervised backpropagation, and visualizes the decision boundary.

rm(list=ls())
graphics.off()
set.seed(1)

# 1. Generate Synthetic XOR-like Data
N <- 200  # total number of samples (must be even)
halfN <- N / 2
X <- matrix(0, nrow = N, ncol = 2)
y <- rep(0, N)

# Class 0: half from around (0,0) and half from around (1,1)
X[1:(halfN/2), ] <- matrix(rep(c(0, 0), each = halfN/2), ncol = 2, byrow = TRUE) + 0.1 * matrix(rnorm((halfN/2) * 2), ncol = 2)
X[(halfN/2 + 1):halfN, ] <- matrix(rep(c(1, 1), each = halfN/2), ncol = 2, byrow = TRUE) + 0.1 * matrix(rnorm((halfN/2) * 2), ncol = 2)
y[1:halfN] <- 0

# Class 1: half from around (0,1) and half from around (1,0)
X[(halfN + 1):(halfN + halfN/2), ] <- matrix(rep(c(0, 1), each = halfN/2), ncol = 2, byrow = TRUE) + 0.1 * matrix(rnorm((halfN/2) * 2), ncol = 2)
X[(halfN + halfN/2 + 1):N, ] <- matrix(rep(c(1, 0), each = halfN/2), ncol = 2, byrow = TRUE) + 0.1 * matrix(rnorm((halfN/2) * 2), ncol = 2)
y[(halfN + 1):N] <- 1

# Plot the data
plot(X, col = ifelse(y == 0, "red", "blue"), pch = 16, xlab = "x1", ylab = "x2", main = "XOR-like Data")

# 2. Pretrain RBM Layer (Unsupervised)
sigmoid <- function(x) { 1 / (1 + exp(-x)) }
rbmTrain <- function(data, num_hidden, learningRate, epochs) {
  n <- nrow(data)
  d_visible <- ncol(data)
  rbm <- list()
  rbm$W <- 0.01 * matrix(rnorm(d_visible * num_hidden), nrow = d_visible)
  rbm$b_visible <- rep(0, d_visible)
  rbm$b_hidden <- rep(0, num_hidden)
  k <- 1
  for (epoch in 1:epochs) {
    for (i in 1:n) {
      v0 <- data[i, ]
      h0_prob <- sigmoid(v0 %*% rbm$W + rbm$b_hidden)
      h0 <- as.numeric(h0_prob > runif(num_hidden))
      vk <- v0
      hk <- h0
      for (step in 1:k) {
        vk_prob <- sigmoid(hk %*% t(rbm$W) + rbm$b_visible)
        vk <- as.numeric(vk_prob > runif(d_visible))
        hk_prob <- sigmoid(vk %*% rbm$W + rbm$b_hidden)
        hk <- as.numeric(hk_prob > runif(num_hidden))
      }
      dW <- (outer(v0, h0_prob)) - (outer(vk, hk_prob))
      db_visible <- v0 - vk
      db_hidden <- h0_prob - hk_prob
      rbm$W <- rbm$W + learningRate * dW
      rbm$b_visible <- rbm$b_visible + learningRate * db_visible
      rbm$b_hidden <- rbm$b_hidden + learningRate * db_hidden
    }
    if (epoch %% 500 == 0) {
      cat(sprintf("RBM Epoch %d complete.\n", epoch))
    }
  }
  rbm
}

rbm <- rbmTrain(X, num_hidden = 12, learningRate = 0.05, epochs = 3000)

rbmTransform <- function(rbm, data) {
  sigmoid(data %*% rbm$W + matrix(rep(rbm$b_hidden, nrow(data)), nrow = nrow(data), byrow = TRUE))
}

H <- rbmTransform(rbm, X)  # Hidden representation

# 3. Train Logistic Regression on RBM Features
logisticTrain <- function(H, y, learningRate, epochs) {
  n <- nrow(H)
  d <- ncol(H)
  W <- 0.01 * matrix(rnorm(d), nrow = d)
  b <- 0
  for (epoch in 1:epochs) {
    scores <- H %*% W + b
    y_pred <- sigmoid(scores)
    loss <- -mean(y * log(y_pred + 1e-15) + (1 - y) * log(1 - y_pred + 1e-15))
    dscores <- y_pred - y
    gradW <- t(H) %*% dscores / n
    gradb <- mean(dscores)
    W <- W - learningRate * gradW
    b <- b - learningRate * gradb
    if (epoch %% 500 == 0) {
      cat(sprintf("Logistic Epoch %d, Loss: %.4f\n", epoch, loss))
    }
  }
  list(W = W, b = b)
}

log_model <- logisticTrain(H, y, learningRate = 0.1, epochs = 3000)
logisticPredict <- function(W, b, H) {
  sigmoid(H %*% W + b)
}

y_pred <- logisticPredict(log_model$W, log_model$b, H)
initAcc <- mean(round(y_pred) == y) * 100
cat(sprintf("Pretrained DBN accuracy: %.2f%%\n", initAcc))

# 4. Fine-Tune the Entire DBN (Supervised Backpropagation)
dbnFineTune <- function(X, y, rbm, W_lr, b_lr, learningRate, epochs) {
  n <- nrow(X)
  lossHistory <- numeric(epochs)
  for (epoch in 1:epochs) {
    Z1 <- X %*% rbm$W + matrix(rep(rbm$b_hidden, n), nrow = n, byrow = TRUE)
    H <- sigmoid(Z1)
    scores <- H %*% W_lr + b_lr
    y_pred <- sigmoid(scores)
    loss <- -mean(y * log(y_pred + 1e-15) + (1 - y) * log(1 - y_pred + 1e-15))
    lossHistory[epoch] <- loss
    dscores <- (y_pred - y) / n
    gradW_lr <- t(H) %*% dscores
    gradb_lr <- sum(dscores)
    dH <- dscores %*% t(W_lr)
    dZ1 <- dH * (H * (1 - H))
    gradW_rbm <- t(X) %*% dZ1
    gradb_rbm <- colMeans(dZ1)
    W_lr <- W_lr - learningRate * gradW_lr
    b_lr <- b_lr - learningRate * gradb_lr
    rbm$W <- rbm$W - learningRate * gradW_rbm
    rbm$b_hidden <- rbm$b_hidden - learningRate * gradb_rbm
    if (epoch %% 200 == 0 || epoch == epochs) {
      cat(sprintf("Fine-Tune Epoch %d, Loss: %.4f\n", epoch, loss))
    }
  }
  list(rbm = rbm, W_lr = W_lr, b_lr = b_lr, lossHistory = lossHistory)
}

ft <- dbnFineTune(X, y, rbm, log_model$W, log_model$b, learningRate = 0.001, epochs = 3000)
rbm <- ft$rbm; W_lr <- ft$W_lr; b_lr <- ft$b_lr
H_ft <- rbmTransform(rbm, X)
y_pred <- logisticPredict(W_lr, b_lr, H_ft)
finalAcc <- mean(round(y_pred) == y) * 100
cat(sprintf("Fine-tuned DBN accuracy: %.2f%%\n", finalAcc))

# 5. Visualize Decision Boundary
decisionBoundary <- function(netFunc, X) {
  margin <- 0.2
  x_min <- min(X[, 1]) - margin
  x_max <- max(X[, 1]) + margin
  y_min <- min(X[, 2]) - margin
  y_max <- max(X[, 2]) + margin
  xGrid <- seq(x_min, x_max, length.out = 100)
  yGrid <- seq(y_min, y_max, length.out = 100)
  gridPoints <- as.matrix(expand.grid(x1 = xGrid, x2 = yGrid))
  preds <- netFunc(gridPoints)
  gridLabels <- matrix(as.numeric(preds >= 0.5), nrow = length(xGrid), ncol = length(yGrid))
  list(xGrid = xGrid, yGrid = yGrid, gridLabels = gridLabels)
}

dbnPredict <- function(x, rbm, W_lr, b_lr) {
  H <- sigmoid(x %*% rbm$W + matrix(rep(rbm$b_hidden, nrow(x)), nrow = nrow(x), byrow = TRUE))
  sigmoid(H %*% W_lr + b_lr)
}

netFunc <- function(x) { dbnPredict(x, rbm, W_lr, b_lr) }
db <- decisionBoundary(netFunc, X)
plot(X, col = ifelse(y == 0, "red", "blue"), pch = 16, xlab = "x1", ylab = "x2", main = "DBN Decision Boundary")
contour(db$xGrid, db$yGrid, db$gridLabels, levels = c(0.5), add = TRUE, lty = 2, lwd = 2)
legend("topright", legend = c("Class 0", "Class 1"), col = c("red", "blue"), pch = 16)
