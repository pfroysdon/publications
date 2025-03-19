# Overfitting vs. Underfitting in Polynomial Regression
#
# This script demonstrates overfitting versus underfitting using polynomial regression.
# Synthetic data is generated from a noisy sine function. Three polynomial models are
# fitted: degree 1 (underfitting), degree 5 (good fit), and degree 15 (overfitting).
# The data, the true function, and fitted curves are plotted, and training MSE is computed.

set.seed(1)

# -----------------------------
# Generate synthetic data
# -----------------------------
N <- 30
x <- seq(-1, 1, length.out = N)
noise <- 0.2 * rnorm(N)
y <- sin(2 * pi * x) + noise

# Define degrees to compare
degrees <- c(1, 5, 15)
x_fine <- seq(-1, 1, length.out = 200)

# Function to build polynomial design matrix
polynomialDesignMatrix <- function(x, degree) {
  X_poly <- outer(x, 0:degree, `^`)
  X_poly
}

# Plot results in subplots
par(mfrow = c(1, 3), mar = c(4,4,2,1))
for (d in degrees) {
  X_design <- polynomialDesignMatrix(x, d)
  w <- solve(t(X_design) %*% X_design) %*% t(X_design) %*% y  # Normal equation
  X_fine_design <- polynomialDesignMatrix(x_fine, d)
  y_pred <- X_fine_design %*% w
  
  # Compute training MSE
  y_train_pred <- X_design %*% w
  mse_train <- mean((y - y_train_pred)^2)
  
  plot(x, y, pch = 16, col = "blue", main = sprintf("Degree %d (MSE = %.3f)", d, mse_train),
       xlab = "x", ylab = "y")
  lines(x_fine, y_pred, col = "red", lwd = 2)
  lines(x_fine, sin(2 * pi * x_fine), lty = 2, col = "black")
  legend("topright", legend = c("Data", sprintf("Degree %d", d), "True Function"), 
         col = c("blue", "red", "black"), pch = c(16, NA, NA), lty = c(NA, 1, 2), bty = "n")
  grid()
}
