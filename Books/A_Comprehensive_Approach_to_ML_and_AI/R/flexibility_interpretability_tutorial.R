# flexibility_interpretability_tutorial.R
# This tutorial demonstrates the trade-off between flexibility and interpretability.
# We generate synthetic data from a non-linear function and fit two models:
#   - A linear model (interpretable but less flexible)
#   - A 5th degree polynomial model (more flexible but less interpretable)
#
# Run this script to see the differences in the fitted curves and the model coefficients.

rm(list = ls())
graphics.off()
set.seed(1)

## 1. Generate synthetic data
N <- 50                              # Number of data points
x <- seq(0, 1, length.out = N)         # Input feature in [0,1]
true_function <- function(x) { sin(2 * pi * x) }  # True underlying non-linear function
noise_std <- 0.1                     # Standard deviation of Gaussian noise
y <- true_function(x) + noise_std * rnorm(N)

## 2. Fit a linear regression model (interpretable, low flexibility)
polynomialDesignMatrix <- function(x, degree) {
  N <- length(x)
  X_poly <- matrix(1, nrow = N, ncol = degree + 1)
  for (d in 1:degree) {
    X_poly[, d + 1] <- x^d
  }
  X_poly
}
degree_linear <- 1
X_linear <- polynomialDesignMatrix(x, degree_linear)
w_linear <- solve(t(X_linear) %*% X_linear) %*% (t(X_linear) %*% y)

## 3. Fit a flexible polynomial regression model (degree 5)
degree_poly <- 5
X_poly <- polynomialDesignMatrix(x, degree_poly)
w_poly <- solve(t(X_poly) %*% X_poly) %*% (t(X_poly) %*% y)

## 4. Evaluate predictions on a fine grid
x_fine <- seq(0, 1, length.out = 200)
X_linear_fine <- polynomialDesignMatrix(x_fine, degree_linear)
y_pred_linear <- X_linear_fine %*% w_linear
X_poly_fine <- polynomialDesignMatrix(x_fine, degree_poly)
y_pred_poly <- X_poly_fine %*% w_poly

## 5. Plot the results
par(mfrow = c(1, 2), mar = c(4,4,3,1))
plot(x, y, col = "blue", pch = 16, xlab = "x", ylab = "y",
     main = "Interpretable Model: Linear Regression")
lines(x_fine, true_function(x_fine), lty = 2, lwd = 2, col = "black")
lines(x_fine, y_pred_linear, lwd = 2, col = "red")
legend("topright", legend = c("True Function", "Linear Fit"), col = c("black", "red"), lty = c(2,1))

plot(x, y, col = "blue", pch = 16, xlab = "x", ylab = "y",
     main = "Flexible Model: 5th Degree Polynomial")
lines(x_fine, true_function(x_fine), lty = 2, lwd = 2, col = "black")
lines(x_fine, y_pred_poly, lwd = 2, col = "red")
legend("topright", legend = c("True Function", "Poly Fit"), col = c("black", "red"), lty = c(2,1))

## 6. Display model coefficients
cat("Linear Regression Coefficients (Interpretable):\n")
print(w_linear)
cat("5th Degree Polynomial Regression Coefficients (Less Interpretable):\n")
print(w_poly)
