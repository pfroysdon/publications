# train_test_split_tutorial.R
# This script generates a synthetic dataset (100 samples, 3 features),
# splits it into training (80%) and testing (20%) sets, and visualizes the split.

set.seed(42)

# Generate synthetic data
X <- matrix(rnorm(100 * 3), nrow = 100, ncol = 3)
Y <- matrix(sample(0:1, 100, replace = TRUE), nrow = 100)

# Define train-test split ratio (80% training, 20% testing)
train_ratio <- 0.8

# Train-test split function
train_test_split <- function(X, Y, train_ratio) {
  N <- nrow(X)
  train_size <- floor(train_ratio * N)
  idx <- sample(1:N)
  X <- X[idx, ]
  Y <- Y[idx, ]
  list(X_train = X[1:train_size, ], Y_train = Y[1:train_size, ],
       X_test = X[(train_size+1):N, ], Y_test = Y[(train_size+1):N, ])
}

split <- train_test_split(X, Y, train_ratio)
X_train <- split$X_train
Y_train <- split$Y_train
X_test  <- split$X_test
Y_test  <- split$Y_test

cat(sprintf("Training set size: %d samples\n", nrow(X_train)))
cat(sprintf("Testing set size: %d samples\n", nrow(X_test)))

# 3D visualization using scatterplot3d
if (!requireNamespace("scatterplot3d", quietly = TRUE)) install.packages("scatterplot3d")
library(scatterplot3d)
scatterplot3d(X_train, color = "blue", pch = 16, main = "Training Data")
scatterplot3d(X_test, color = "red", pch = 16, main = "Testing Data")
