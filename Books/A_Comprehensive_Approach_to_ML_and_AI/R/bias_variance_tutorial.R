# bias_variance_decomposition_tutorial.R
# This script demonstrates bias-variance decomposition.
# It generates M training sets from a noisy sine function, fits a polynomial regression model (degree 3),
# and computes the squared bias, variance, and total expected error on a fine grid.

bias_variance_decomposition <- function() {
  set.seed(0)
  M <- 100          # Number of experiments
  N <- 30           # Points per training set
  degree <- 3       # Fixed polynomial degree
  noise_std <- 0.2
  
  true_function <- function(x) sin(2 * pi * x)
  x_fine <- seq(-1, 1, length.out = 200)
  y_true <- true_function(x_fine)
  
  predictions <- matrix(0, nrow = length(x_fine), ncol = M)
  
  polynomialDesignMatrix <- function(x, degree) {
    sapply(0:degree, function(d) x^d)
  }
  
  for (m in 1:M) {
    x_train <- runif(N, min = -1, max = 1)
    y_train <- true_function(x_train) + noise_std * rnorm(N)
    X_train <- polynomialDesignMatrix(x_train, degree)
    w <- solve(t(X_train) %*% X_train, t(X_train) %*% y_train)
    X_fine <- polynomialDesignMatrix(x_fine, degree)
    predictions[, m] <- X_fine %*% w
  }
  
  avg_prediction <- rowMeans(predictions)
  bias_sq <- (avg_prediction - y_true)^2
  variance <- apply(predictions, 1, var)
  noise_variance <- noise_std^2
  total_error <- bias_sq + variance + noise_variance
  
  par(mfrow = c(2,1))
  plot(x_fine, y_true, type = "l", col = "black", lwd = 2, ylim = range(c(y_true, predictions)),
       xlab = "x", ylab = "y", main = "True Function and Predictions")
  lines(x_fine, avg_prediction, col = "red", lwd = 2)
  for (m in 1:10) {
    lines(x_fine, predictions[, m], col = "blue", lwd = 1)
  }
  legend("topright", legend = c("True", "Average"), col = c("black", "red"), lty = 1)
  grid()
  
  plot(x_fine, bias_sq, type = "l", col = "red", lwd = 2, ylim = range(c(bias_sq, variance, total_error)),
       xlab = "x", ylab = "Error", main = "Error Decomposition")
  lines(x_fine, variance, col = "blue", lwd = 2)
  lines(x_fine, total_error, col = "black", lwd = 2)
  legend("topright", legend = c("Bias^2", "Variance", "Total Error"), col = c("red", "blue", "black"), lty = 1)
  grid()
  
  cat(sprintf("Average Bias^2: %.4f\n", mean(bias_sq)))
  cat(sprintf("Average Variance: %.4f\n", mean(variance)))
  cat(sprintf("Average Total Error (including noise): %.4f\n", mean(total_error)))
}

bias_variance_decomposition()
