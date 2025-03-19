# Clear workspace and close figures
rm(list = ls())
graphics.off()

# ------------------------------
# Generate Mixture Data
# ------------------------------
set.seed(123)
# 100 samples from N(0,1) and 100 samples from N(3,1)
X <- c(rnorm(100, mean = 0, sd = 1), rnorm(100, mean = 3, sd = 1))

# ------------------------------
# Define Kernel Density Estimation function (from scratch)
# ------------------------------
myKDE <- function(X, h, num_points = 100) {
  # Create a grid of evaluation points
  x_min <- min(X) - 3 * h
  x_max <- max(X) + 3 * h
  x_grid <- seq(x_min, x_max, length.out = num_points)
  
  # Define Gaussian kernel function
  K <- function(u) (1 / sqrt(2 * pi)) * exp(-0.5 * u^2)
  
  n <- length(X)
  f_hat <- numeric(length(x_grid))
  
  # Compute the KDE by summing kernel contributions at each grid point
  for (i in seq_along(x_grid)) {
    u <- (x_grid[i] - X) / h
    f_hat[i] <- sum(K(u))
  }
  f_hat <- f_hat / (n * h)
  
  return(list(x_grid = x_grid, f_hat = f_hat))
}

# Use the function with h = 0.5 and 200 evaluation points
kde_result <- myKDE(X, h = 0.5, num_points = 200)

# Plot the estimated density
plot(kde_result$x_grid, kde_result$f_hat, type = "l", lwd = 2,
     main = "Kernel Density Estimate for a Mixture of Gaussians",
     xlab = "x", ylab = "Estimated Density")
