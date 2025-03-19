# Clear workspace and close figures
rm(list = ls())
graphics.off()

set.seed(42)
# ------------------------------
# Generate Synthetic Dataset
# ------------------------------
# Create two Gaussian clusters in 2D
X <- rbind(matrix(rnorm(50 * 2), ncol = 2) + 2,
           matrix(rnorm(50 * 2), ncol = 2) - 2)
# Define class labels (first 50 are 1, next 50 are 0)
Y <- c(rep(1, 50), rep(0, 50))

# ------------------------------
# Split Data into Training and Testing Sets
# ------------------------------
train_ratio <- 0.8
train_size <- floor(train_ratio * nrow(X))
X_train <- X[1:train_size, ]
Y_train <- Y[1:train_size]
X_test <- X[(train_size + 1):nrow(X), ]
Y_test <- Y[(train_size + 1):length(Y)]

# ------------------------------
# Define Optimized KNN Function (fast version)
# ------------------------------
knn_classify_fast <- function(X_train, Y_train, X_test, K) {
  num_test <- nrow(X_test)
  Y_pred <- numeric(num_test)
  
  for (i in 1:num_test) {
    # Compute Euclidean distances from test point to all training points
    distances <- sqrt(rowSums((X_train - matrix(X_test[i, ], nrow = nrow(X_train), ncol = ncol(X_train), byrow = TRUE))^2))
    idx <- order(distances)
    nearest_labels <- Y_train[idx[1:K]]
    # Use the mode (most frequent class)
    Y_pred[i] <- as.numeric(names(which.max(table(nearest_labels))))
  }
  return(Y_pred)
}

# ------------------------------
# Apply KNN (K = 5)
# ------------------------------
K <- 5
Y_pred <- knn_classify_fast(X_train, Y_train, X_test, K)

# Compute and display accuracy
accuracy <- mean(Y_pred == Y_test) * 100
cat(sprintf("KNN Accuracy: %.2f%%\n", accuracy))

# ------------------------------
# Create a Mesh Grid for Visualization
# ------------------------------
x1_range <- seq(min(X[,1]), max(X[,1]), length.out = 100)
x2_range <- seq(min(X[,2]), max(X[,2]), length.out = 100)
grid <- expand.grid(x1 = x1_range, x2 = x2_range)
X_grid <- as.matrix(grid)

# Predict on the grid
Y_grid <- knn_classify_fast(X_train, Y_train, X_grid, K)
# Reshape predictions to a matrix for contour plotting
Y_grid_matrix <- matrix(Y_grid, nrow = length(x1_range), ncol = length(x2_range))

# ------------------------------
# Plot Decision Boundary and Training Data
# ------------------------------
# First plot: scatter plot of training data with colors according to class
plot(X_train, col = ifelse(Y_train == 1, "blue", "red"), pch = 16,
     xlab = "Feature 1", ylab = "Feature 2", main = "KNN Clustering")
legend("topright", legend = c("Class 1", "Class 0"), col = c("blue", "red"), pch = 16)

# Second plot: include the decision boundary via contour
plot(X_train, col = ifelse(Y_train == 1, "blue", "red"), pch = 16,
     xlab = "Feature 1", ylab = "Feature 2", main = "KNN Clustering - Decision Boundary")
contour(x1_range, x2_range, Y_grid_matrix, add = TRUE, drawlabels = FALSE, lwd = 0.8)
legend("topright", legend = c("Class 1", "Class 0"), col = c("blue", "red"), pch = 16)
