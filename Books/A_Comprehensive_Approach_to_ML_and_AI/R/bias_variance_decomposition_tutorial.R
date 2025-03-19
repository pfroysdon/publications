# bias_variance_tutorial.R
# This tutorial demonstrates the bias-variance trade-off using polynomial regression.
# It fits polynomials of degrees 1 to 10 and plots training and test MSE,
# as well as visualizing a sample fit.

bias_variance_tutorial <- function() {
  set.seed(0)
  N <- 50
  x <- seq(-1, 1, length.out = N)
  true_function <- function(x) sin(2 * pi * x)
  noise_std <- 0.2
  y <- true_function(x) + noise_std * rnorm(N)
  
  Ntrain <- round(0.7 * N)
  x_train <- x[1:Ntrain]
  y_train <- y[1:Ntrain]
  x_test <- x[(Ntrain + 1):N]
  y_test <- y[(Ntrain + 1):N]
  
  max_degree <- 10
  train_errors <- numeric(max_degree)
  test_errors <- numeric(max_degree)
  
  polynomialDesignMatrix <- function(x, degree) {
    sapply(0:degree, function(d) x^d)
  }
  
  for (d in 1:max_degree) {
    X_train <- polynomialDesignMatrix(x_train, d)
    w <- solve(t(X_train) %*% X_train, t(X_train) %*% y_train)
    y_pred_train <- X_train %*% w
    train_errors[d] <- mean((y_train - y_pred_train)^2)
    
    X_test <- polynomialDesignMatrix(x_test, d)
    y_pred_test <- X_test %*% w
    test_errors[d] <- mean((y_test - y_pred_test)^2)
  }
  
  par(mfrow = c(1,2))
  plot(1:max_degree, train_errors, type = "b", col = "blue", pch = 16,
       xlab = "Polynomial Degree", ylab = "MSE", main = "Training and Test Errors")
  lines(1:max_degree, test_errors, type = "b", col = "red", pch = 16)
  legend("topright", legend = c("Training Error", "Test Error"), col = c("blue", "red"), pch = 16)
  grid()
  
  d_select <- 5
  X_train_sel <- polynomialDesignMatrix(x_train, d_select)
  w_sel <- solve(t(X_train_sel) %*% X_train_sel, t(X_train_sel) %*% y_train)
  x_fine <- seq(-1, 1, length.out = 200)
  X_fine_sel <- polynomialDesignMatrix(x_fine, d_select)
  y_pred_fine <- X_fine_sel %*% w_sel
  
  plot(x_train, y_train, pch = 16, col = "blue", xlab = "x", ylab = "y",
       main = sprintf("Polynomial Fit (Degree = %d)", d_select))
  lines(x_fine, true_function(x_fine), lty = 2, col = "black", lwd = 2)
  lines(x_fine, y_pred_fine, col = "red", lwd = 2)
  legend("topright", legend = c("Training Data", "True Function", "Prediction"),
         col = c("blue", "black", "red"), lty = c(NA, 2, 1), pch = c(16, NA, NA))
  grid()
}

bias_variance_tutorial()
