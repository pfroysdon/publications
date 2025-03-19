# svm_tutorial.R
# SVM Tutorial in R (using a custom QP-based implementation)
# Generates synthetic data for two classes and trains SVMs with a linear kernel
# and an RBF kernel. Decision boundaries are visualized.
# Note: This example uses the 'quadprog' package for solving QP problems.

set.seed(42)
library(quadprog)

# Generate synthetic dataset
num_samples <- 100
# Class +1: centered at (2,2)
X1 <- matrix(rnorm(num_samples/2 * 2), ncol = 2) + 2
# Class -1: centered at (-2,-2)
X2 <- matrix(rnorm(num_samples/2 * 2), ncol = 2) - 2
X <- rbind(X1, X2)  # Feature matrix (100x2)
Y <- c(rep(1, num_samples/2), rep(-1, num_samples/2))  # Labels (100x1)

# Shuffle the data
idx <- sample(1:num_samples)
X <- X[idx, ]
Y <- Y[idx]

# Plot dataset
plot(X[Y==1, 1], X[Y==1, 2], col = "blue", pch = 16, xlab = "Feature 1", ylab = "Feature 2",
     main = "SVM Classification")
points(X[Y==-1, 1], X[Y==-1, 2], col = "red", pch = 16)
legend("topright", legend = c("Class +1", "Class -1"), col = c("blue", "red"), pch = 16)
grid()

# SVM parameters
C <- 1          # Regularization parameter
sigma <- 0.5    # RBF kernel width

# Function: SVM Training with Kernel
svm_train_kernel <- function(X, Y, C, kernel = "linear", sigma) {
  N <- nrow(X)
  # Compute kernel matrix K (NxN)
  K <- matrix(0, nrow = N, ncol = N)
  for (i in 1:N) {
    for (j in 1:N) {
      if (kernel == "rbf") {
        K[i,j] <- exp(-sum((X[i,] - X[j,])^2) / (2 * sigma^2))
      } else if (kernel == "linear") {
        K[i,j] <- sum(X[i,] * X[j,])
      }
    }
  }
  
  # Set up quadratic programming: minimize (1/2)alpha' * H * alpha - 1' * alpha
  H <- (Y %*% t(Y)) * K
  f <- -rep(1, N)
  Aeq <- matrix(Y, nrow = 1)
  beq <- 0
  lb <- rep(0, N)
  ub <- rep(C, N)
  
  # Solve QP: quadprog::solve.QP requires Dmat to be positive definite.
  sol <- solve.QP(Dmat = H + 1e-8*diag(N), dvec = f, Amat = t(rbind(Aeq, diag(N))), 
                   bvec = c(beq, lb), meq = 1)
  alpha <- sol$solution
  
  # Compute weight vector for linear kernel if applicable
  if (kernel == "linear") {
    W <- colSums(matrix(alpha * Y, nrow = N, ncol = ncol(X)) * X)
  } else {
    W <- NULL
  }
  
  # Compute bias using support vectors (alpha > threshold)
  sv_idx <- which(alpha > 1e-4)
  b <- mean(Y[sv_idx] - if (!is.null(W)) {X[sv_idx, ] %*% W} else {
    sapply(sv_idx, function(i) {
      sum(alpha * Y * sapply(1:N, function(j) {
        if (kernel == "rbf") exp(-sum((X[i,]-X[j,])^2)/(2*sigma^2)) else sum(X[i,]*X[j,])
      }))
    })
  })
  
  list(W = W, b = b, alpha = alpha, X_train = X, Y_train = Y, kernel = kernel, sigma = sigma)
}

# Function: SVM Prediction with Kernel
svm_predict_kernel <- function(model, X_test) {
  N_test <- nrow(X_test)
  N_train <- nrow(model$X_train)
  K_test <- matrix(0, nrow = N_test, ncol = N_train)
  for (i in 1:N_test) {
    for (j in 1:N_train) {
      if (model$kernel == "rbf") {
        K_test[i,j] <- exp(-sum((X_test[i,] - model$X_train[j,])^2) / (2 * model$sigma^2))
      } else if (model$kernel == "linear") {
        K_test[i,j] <- sum(X_test[i,] * model$X_train[j,])
      }
    }
  }
  # Prediction: f(x) = sum_j alpha_j*y_j*K(x, x_j) + b
  f_x <- K_test %*% (model$alpha * model$Y_train) + model$b
  sign(f_x)
}

# Train SVM with linear kernel
model_linear <- svm_train_kernel(X, Y, C, kernel = "linear", sigma = sigma)
cat("Linear Kernel SVM Training Complete.\n")

# Train SVM with RBF kernel
model_rbf <- svm_train_kernel(X, Y, C, kernel = "rbf", sigma = sigma)
cat("RBF Kernel SVM Training Complete.\n")

# Predict on training data
Y_pred_linear <- svm_predict_kernel(model_linear, X)
accuracy_linear <- mean(Y_pred_linear == Y) * 100
cat(sprintf("Linear SVM Accuracy: %.2f%%\n", accuracy_linear))

Y_pred_rbf <- svm_predict_kernel(model_rbf, X)
accuracy_rbf <- mean(Y_pred_rbf == Y) * 100
cat(sprintf("RBF SVM Accuracy: %.2f%%\n", accuracy_rbf))

# Function to plot decision boundary
plot_decision_boundary <- function(model, X, Y, kernel, sigma) {
  x1_grid <- seq(min(X[,1])-1, max(X[,1])+1, length.out = 100)
  x2_grid <- seq(min(X[,2])-1, max(X[,2])+1, length.out = 100)
  grid <- expand.grid(x1 = x1_grid, x2 = x2_grid)
  preds <- svm_predict_kernel(model, as.matrix(grid))
  Z <- matrix(preds, nrow = 100, ncol = 100)
  
  # Plot the data and decision boundary
  plot(X[Y==1,1], X[Y==1,2], col = "blue", pch = 16, xlab = "Feature 1", ylab = "Feature 2",
       main = paste("SVM -", toupper(kernel), "Kernel Decision Boundary"))
  points(X[Y==-1,1], X[Y==-1,2], col = "red", pch = 16)
  contour(x1_grid, x2_grid, Z, levels = 0, add = TRUE, lwd = 2, col = "black")
  legend("topright", legend = c("Class +1", "Class -1"), col = c("blue", "red"), pch = 16)
  grid()
}

# Plot decision boundaries
plot_decision_boundary(model_linear, X, Y, "linear", sigma)
plot_decision_boundary(model_rbf, X, Y, "rbf", sigma)
