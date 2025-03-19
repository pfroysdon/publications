# Clear workspace and close figures
rm(list = ls())
graphics.off()

# ------------------------------
# Generate Synthetic Data (Position Estimation)
# ------------------------------
N <- 10                    # Number of measurements
true_position <- 5         # True state

# Observation matrix (direct observation: all ones)
H <- matrix(1, nrow = N, ncol = 1)

# Generate noisy measurements (add Gaussian noise with sd = 0.5)
set.seed(123)
Y <- true_position + rnorm(N, mean = 0, sd = 0.5)

# ------------------------------
# Define the Least Squares Filter Function
# ------------------------------
least_squares_filter <- function(H, Y) {
  # Normal Equation: X_est = (H' H)^{-1} H' Y
  X_est <- solve(t(H) %*% H) %*% t(H) %*% Y
  return(as.numeric(X_est))
}

# Apply the Least Squares Filter
X_est <- least_squares_filter(H, Y)

# Display the true and estimated positions
cat(sprintf("True Position: %.2f\n", true_position))
cat(sprintf("Estimated Position: %.2f\n", X_est))

# ------------------------------
# Plot the Measurements and Estimates
# ------------------------------
plot(1:N, Y, col = "red", pch = 16, xlab = "Measurement Index", ylab = "Position Estimate",
     main = "Least Squares Filter Estimation")
lines(1:N, rep(X_est, N), col = "blue", lwd = 2)  # LSF estimate
lines(1:N, rep(true_position, N), col = "black", lty = 2, lwd = 2)  # True position
legend("topright", legend = c("Noisy Measurements", "LSF Estimate", "True Position"),
       col = c("red", "blue", "black"), pch = c(16, NA, NA), lty = c(NA, 1, 2), lwd = c(NA, 2, 2))
