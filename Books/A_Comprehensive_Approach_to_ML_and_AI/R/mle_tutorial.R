# MLE Example – Estimate Parameters of a Normal Distribution using MLE
#
# Generate synthetic data from a normal distribution with true parameters.
true_mu <- 5
true_sigma <- 2
n <- 1000
data <- true_mu + true_sigma * rnorm(n)

# Compute MLE estimates
mu_hat <- mean(data)
sigma2_hat <- mean((data - mu_hat)^2)
sigma_hat <- sqrt(sigma2_hat)

cat(sprintf("Estimated Mean: %.4f\n", mu_hat))
cat(sprintf("Estimated Std Dev: %.4f\n", sigma_hat))

# Plot histogram of data and fitted normal density
hist(data, breaks = 30, probability = TRUE, col = "lightgray",
     main = "MLE for Normal Distribution", xlab = "Data Value", ylab = "Probability Density")
x_values <- seq(min(data), max(data), length.out = 100)
y_values <- dnorm(x_values, mean = mu_hat, sd = sigma_hat)
lines(x_values, y_values, col = "red", lwd = 2)
legend("topright", legend = c("Data Histogram", "Fitted Normal PDF"),
       col = c("lightgray", "red"), lty = c(1, 1), lwd = c(NA, 2))
grid()
