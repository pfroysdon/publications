# vnn_simple_tutorial.R
# VNN Tutorial for Binary Classification from Scratch in R
# Generates synthetic data for two classes, trains a one-hidden-layer neural network
# using gradient descent, computes training accuracy, and plots the decision boundary.

set.seed(1)

# Generate Synthetic Data
N <- 100
# Class +1: centered at (2,2)
X1 <- matrix(rnorm(N * 2), ncol = 2) + 2
# Class -1: centered at (-2,-2)
X2 <- matrix(rnorm(N * 2), ncol = 2) - 2
X <- rbind(X1, X2)
y <- c(rep(1, N), rep(-1, N))

# Train a simple VNN from scratch
hiddenSize <- 10
learningRate <- 0.02
epochs <- 500

# Define activation functions
relu <- function(x) pmax(x, 0)
tanh_act <- function(x) tanh(x)

# Train VNN using gradient descent (one hidden layer)
VNNTrain <- function(X, y, hiddenSize, learningRate, epochs) {
  n <- nrow(X); d <- ncol(X)
  # Initialize weights and biases
  W1 <- matrix(rnorm(hiddenSize * d, sd = 0.01), nrow = hiddenSize)
  b1 <- matrix(0, nrow = hiddenSize, ncol = 1)
  W2 <- matrix(rnorm(1 * hiddenSize, sd = 0.01), nrow = 1)
  b2 <- 0
  lossHistory <- numeric(epochs)
  
  for (epoch in 1:epochs) {
    # Forward pass
    Z1 <- W1 %*% t(X) + matrix(rep(b1, n), nrow = hiddenSize)
    A1 <- relu(Z1)
    Z2 <- W2 %*% A1 + b2
    A2 <- tanh_act(Z2)
    loss <- mean(0.5 * (A2 - t(y))^2)
    lossHistory[epoch] <- loss
    
    # Backpropagation
    dA2 <- (A2 - t(y)) / n
    dZ2 <- dA2 * (1 - A2^2)  # derivative of tanh
    dW2 <- dZ2 %*% t(A1)
    db2 <- sum(dZ2)
    dA1 <- t(W2) %*% dZ2
    dZ1 <- dA1
    dZ1[Z1 <= 0] <- 0
    dW1 <- dZ1 %*% X
    db1 <- rowMeans(dZ1)
    
    # Parameter updates
    W1 <- W1 - learningRate * dW1
    b1 <- b1 - learningRate * matrix(db1, ncol = 1)
    W2 <- W2 - learningRate * dW2
    b2 <- b2 - learningRate * db2
  }
  list(W1 = W1, b1 = b1, W2 = W2, b2 = b2, lossHistory = lossHistory)
}

VNNPredict <- function(model, X) {
  n <- nrow(X)
  Z1 <- model$W1 %*% t(X) + matrix(rep(model$b1, n), nrow = nrow(model$W1))
  A1 <- relu(Z1)
  Z2 <- model$W2 %*% A1 + model$b2
  A2 <- tanh_act(Z2)
  as.vector(A2)
}

model <- VNNTrain(X, y, hiddenSize, learningRate, epochs)
y_pred <- VNNPredict(model, X)
accuracy <- mean(sign(y_pred) == y) * 100
cat(sprintf("Training Accuracy: %.2f%%\n", accuracy))

# Visualize Decision Boundary
x1_range <- seq(min(X[,1])-1, max(X[,1])+1, length.out = 100)
x2_range <- seq(min(X[,2])-1, max(X[,2])+1, length.out = 100)
grid <- expand.grid(x1 = x1_range, x2 = x2_range)
predictions <- VNNPredict(model, as.matrix(grid))
Z <- matrix(sign(predictions), nrow = 100, ncol = 100)

# Plot data and boundary
plot(X[y==1,1], X[y==1,2], col = "blue", pch = 16, xlab = "Feature 1", ylab = "Feature 2",
     main = "VNN Decision Boundary")
points(X[y==-1,1], X[y==-1,2], col = "red", pch = 16)
contour(x1_range, x2_range, Z, levels = 0, add = TRUE, lwd = 2, col = "black")
legend("topright", legend = c("Class +1", "Class -1"), col = c("blue", "red"), pch = 16)
grid()

# Plot loss history
plot(model$lossHistory, type = "l", lwd = 2, xlab = "Epoch", ylab = "MSE Loss",
     main = "Training Loss History")
grid()
