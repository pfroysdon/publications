# Mixture of Experts (MoE) Tutorial in R
#
# This tutorial demonstrates how to implement a Mixture of Experts (MoE) model
# from scratch to solve a regression problem. The model consists of:
#   - Two expert models (simple linear regressors)
#   - A gating network that computes softmax weights for the experts
#
# For an input x, each expert produces a prediction f_i(x) and the gating network outputs
# weights g_i(x) (which sum to 1). The overall prediction is:
#       y_hat = sum_i g_i(x) * f_i(x)
#
# The model is trained using stochastic gradient descent to minimize the mean-squared
# error (MSE) loss.

set.seed(1)

# -----------------------------
# Generate Synthetic Data
# -----------------------------
N <- 200
X <- matrix(seq(0, 4, length.out = N), ncol = 1)  # 200 x 1 inputs
noise <- 0.5 * rnorm(N)
y <- ifelse(X <= 2, 2 * X, 8 - 2 * X) + noise

# Plot the synthetic data
plot(X, y, pch = 16, col = "blue", main = "Synthetic Data for MoE Regression",
     xlab = "x", ylab = "y")
grid()

# -----------------------------
# Define Helper Functions
# -----------------------------
softmax <- function(z) {
  z <- z - max(z)
  exp(z) / sum(exp(z))
}

# -----------------------------
# MoE Training Function
# -----------------------------
moeTrain <- function(X, y, K, learningRate, epochs) {
  n <- nrow(X); d <- ncol(X)
  
  # Initialize gating network parameters: weights (K x d) and biases (K x 1)
  gating_W <- matrix(rnorm(K * d, sd = 0.01), nrow = K)
  gating_b <- matrix(0, nrow = K, ncol = 1)
  
  # Initialize expert parameters (linear models): weights (K x d) and biases (K x 1)
  expert_W <- matrix(rnorm(K * d, sd = 0.01), nrow = K)
  expert_b <- matrix(0, nrow = K, ncol = 1)
  
  for (epoch in 1:epochs) {
    # Shuffle data
    idx <- sample(1:n)
    totalLoss <- 0
    for (i in idx) {
      x <- matrix(X[i, ], ncol = 1)  # d x 1 vector
      target <- y[i]
      
      # Forward pass for gating network: s = W*x + b, then softmax
      s <- gating_W %*% x + gating_b  # K x 1
      g <- softmax(s)
      
      # Experts: each expert predicts f_i = expert_W[i,]*x + expert_b[i]
      f <- gating_W <- NULL  # not used; we compute expert predictions directly:
      f <- expert_W %*% x + expert_b  # K x 1
      
      # Overall prediction: weighted sum
      y_hat <- sum(g * f)
      
      # Loss: squared error
      e <- y_hat - target
      loss <- 0.5 * e^2
      totalLoss <- totalLoss + loss
      
      # Backward pass:
      # Gradients for expert parameters
      dW_expert <- e * (g %*% t(x))  # K x d
      db_expert <- e * g             # K x 1
      
      # Gradients for gating parameters:
      # For gating, gradient d/ds_i = e * g_i * (f_i - y_hat)
      d_s <- e * (g * (f - y_hat))
      dW_gating <- d_s %*% t(x)
      db_gating <- d_s
      
      # Parameter updates (SGD)
      expert_W <- expert_W - learningRate * dW_expert
      expert_b <- expert_b - learningRate * db_expert
      gating_W <- gating_W - learningRate * dW_gating
      gating_b <- gating_b - learningRate * db_gating
    }
    if (epoch %% 1000 == 0) {
      cat(sprintf("Epoch %d, Loss: %.4f\n", epoch, totalLoss/n))
    }
  }
  
  list(gating = list(W = gating_W, b = gating_b),
       expert = list(W = expert_W, b = expert_b))
}

# -----------------------------
# MoE Prediction Functions
# -----------------------------
moePredict <- function(model, X) {
  n <- nrow(X)
  y_pred <- numeric(n)
  for (i in 1:n) {
    x <- matrix(X[i, ], ncol = 1)
    s <- model$gating$W %*% x + model$gating$b
    g <- softmax(s)
    f <- model$expert$W %*% x + model$expert$b
    y_pred[i] <- sum(g * f)
  }
  y_pred
}

moePredictGated <- function(model, x) {
  # For a single input x (d x 1), return overall prediction, expert outputs, and gating weights.
  s <- model$gating$W %*% x + model$gating$b
  gating_out <- softmax(s)
  expert_out <- model$expert$W %*% x + model$expert$b
  y_hat <- sum(gating_out * expert_out)
  list(y_hat = y_hat, expert_out = expert_out, gating_out = gating_out)
}

# -----------------------------
# Train MoE Model and Evaluate
# -----------------------------
K <- 2                # Number of experts
learningRate <- 0.01
epochs <- 10000

model <- moeTrain(X, y, K, learningRate, epochs)
y_pred <- moePredict(model, X)

# Plot true vs. predicted outputs
plot(X, y, pch = 16, col = "blue", main = "MoE: True Data vs. Prediction",
     xlab = "x", ylab = "y")
lines(X, y_pred, col = "red", lwd = 2)
legend("topright", legend = c("True Data", "Prediction"), col = c("blue", "red"), lty = c(NA,1))

# Visualize experts and gating network outputs for each input
N <- nrow(X)
expert_preds <- matrix(0, nrow = N, ncol = K)
gating_weights <- matrix(0, nrow = N, ncol = K)
for (i in 1:N) {
  x_val <- matrix(X[i, ], ncol = 1)
  result <- moePredictGated(model, x_val)
  expert_preds[i, ] <- as.vector(result$expert_out)
  gating_weights[i, ] <- as.vector(result$gating_out)
}

# Plot experts' predictions
plot(X, expert_preds[,1], type = "l", col = "green", lwd = 2,
     xlab = "x", ylab = "Expert Prediction", main = "Experts' Predictions")
lines(X, expert_preds[,2], col = "magenta", lwd = 2)
legend("topright", legend = c("Expert 1", "Expert 2"), col = c("green", "magenta"), lty = 1)

# Plot gating network outputs
plot(X, gating_weights[,1], type = "l", col = "green", lwd = 2,
     xlab = "x", ylab = "Gating Weight", main = "Gating Network Outputs")
lines(X, gating_weights[,2], col = "magenta", lwd = 2)
legend("topright", legend = c("Weight Expert 1", "Weight Expert 2"), col = c("green", "magenta"), lty = 1)
