# MLP Tutorial in R – Multi–Layer Perceptron for Binary Classification
#
# Generate a synthetic dataset with two classes, train an MLP with one hidden layer,
# and visualize the decision boundary.

set.seed(42)
num_samples <- 200

# Class 1 (label 0)
X1 <- matrix(rnorm((num_samples/2) * 2), ncol = 2) + 1
Y1 <- rep(0, num_samples/2)

# Class 2 (label 1)
X2 <- matrix(rnorm((num_samples/2) * 2), ncol = 2) - 1
Y2 <- rep(1, num_samples/2)

# Combine and shuffle dataset
X <- rbind(X1, X2)
Y <- c(Y1, Y2)
idx <- sample(1:num_samples)
X <- X[idx, ]
Y <- Y[idx]

# Plot the dataset
plot(X[Y == 0, 1], X[Y == 0, 2], col = "red", pch = 16,
     xlab = "Feature 1", ylab = "Feature 2", main = "Synthetic Binary Classification Dataset")
points(X[Y == 1, 1], X[Y == 1, 2], col = "blue", pch = 16)
legend("topright", legend = c("Class 0", "Class 1"), col = c("red", "blue"), pch = 16)
grid()

# Normalize input features
X <- scale(X)

# Define MLP parameters
num_hidden <- 5   # Number of hidden neurons
alpha <- 0.1      # Learning rate
epochs <- 1000    # Training iterations

# MLP Training Function
mlp_train <- function(X, Y, num_hidden, alpha, epochs) {
  N <- nrow(X)
  input_dim <- ncol(X)
  
  # He initialization for weights
  W1 <- matrix(rnorm(input_dim * num_hidden, sd = sqrt(2 / input_dim)), nrow = input_dim)
  b1 <- rep(0, num_hidden)
  W2 <- matrix(rnorm(num_hidden, sd = sqrt(2 / num_hidden)), nrow = num_hidden)
  b2 <- 0
  
  for (epoch in 1:epochs) {
    # Forward propagation
    Z1 <- X %*% W1 + matrix(rep(b1, each = N), nrow = N)
    A1 <- pmax(Z1, 0)  # ReLU activation
    Z2 <- A1 %*% W2 + b2
    A2 <- 1 / (1 + exp(-Z2))  # Sigmoid activation
    
    # Compute binary cross–entropy loss
    loss <- -mean(Y * log(A2 + 1e-8) + (1 - Y) * log(1 - A2 + 1e-8))
    
    # Backpropagation
    dZ2 <- A2 - Y
    dW2 <- t(A1) %*% dZ2 / N
    db2 <- mean(dZ2)
    dA1 <- dZ2 %*% t(W2)
    dZ1 <- dA1 * (Z1 > 0)  # ReLU derivative
    dW1 <- t(X) %*% dZ1 / N
    db1 <- colMeans(dZ1)
    
    # Update parameters
    W1 <- W1 - alpha * dW1
    b1 <- b1 - alpha * db1
    W2 <- W2 - alpha * dW2
    b2 <- b2 - alpha * db2
    
    if (epoch %% 100 == 0) {
      cat(sprintf("Epoch %d, Loss: %.4f\n", epoch, loss))
    }
  }
  
  list(W1 = W1, b1 = b1, W2 = W2, b2 = b2)
}

# MLP Prediction Function
mlp_predict <- function(model, X) {
  N <- nrow(X)
  Z1 <- X %*% model$W1 + matrix(rep(model$b1, each = N), nrow = N)
  A1 <- pmax(Z1, 0)
  Z2 <- A1 %*% model$W2 + model$b2
  A2 <- 1 / (1 + exp(-Z2))
  ifelse(A2 >= 0.5, 1, 0)
}

# Train the MLP
model <- mlp_train(X, Y, num_hidden, alpha, epochs)
cat("Training complete!\n")

# Predict on training data and compute accuracy
Y_pred <- mlp_predict(model, X)
accuracy <- mean(Y_pred == Y) * 100
cat(sprintf("Model Accuracy: %.2f%%\n", accuracy))

# Create a mesh grid for visualization
x1_range <- seq(min(X[, 1]), max(X[, 1]), length.out = 100)
x2_range <- seq(min(X[, 2]), max(X[, 2]), length.out = 100)
grid <- expand.grid(x1 = x1_range, x2 = x2_range)
X_grid <- as.matrix(grid)

# Predict labels for the grid
Y_grid <- mlp_predict(model, X_grid)
Y_grid_matrix <- matrix(Y_grid, nrow = 100, ncol = 100)

# Plot decision boundary
plot(X[Y == 0, 1], X[Y == 0, 2], col = "red", pch = 16,
     xlab = "Feature 1", ylab = "Feature 2", main = "MLP Decision Boundary")
points(X[Y == 1, 1], X[Y == 1, 2], col = "blue", pch = 16)
contour(x1_range, x2_range, Y_grid_matrix, add = TRUE, lwd = 0.8, drawlabels = FALSE, col = "black")
legend("topright", legend = c("Class 0", "Class 1"), col = c("red", "blue"), pch = 16)
grid()
