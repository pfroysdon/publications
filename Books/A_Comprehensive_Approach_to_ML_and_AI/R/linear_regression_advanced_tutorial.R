# Clear workspace and close figures
rm(list = ls())
graphics.off()

# ------------------------------
# Define the Data
# ------------------------------
x <- 1:100
y <- 1.07e-4 * x^3 - 0.0088 * x^2 + 0.3 * x + 2.1

# Create noisy observations (y_tilde) by adding Gaussian noise (sigma = 2)
set.seed(1)
sigma <- 2
y_tilde <- y + rnorm(length(x), mean = 0, sd = sigma)

# Plot the raw data and the true function
plot(x, y, type = "b", col = "black", lwd = 1, xlab = "Time (day)", ylab = "Stock Value ($)", main = "Raw Data")
points(x, y_tilde, col = "blue", pch = 1)
legend("topright", legend = c("True Data", "Noisy Data"), col = c("black", "blue"), pch = c(1, 1))

# ------------------------------
# Fit Models Using Least Squares
# ------------------------------
# First-order (linear) fit
model1 <- lm(y_tilde ~ x)
f1 <- predict(model1, data.frame(x = x))
e1 <- sqrt(sum((y_tilde - f1)^2))  # Euclidean norm of the error

# Second-order (quadratic) fit
model2 <- lm(y_tilde ~ poly(x, 2, raw = TRUE))
f2 <- predict(model2, data.frame(x = x))
e2 <- sqrt(sum((y_tilde - f2)^2))

# Third-order (cubic) fit
model3 <- lm(y_tilde ~ poly(x, 3, raw = TRUE))
f3 <- predict(model3, data.frame(x = x))
e3 <- sqrt(sum((y_tilde - f3)^2))

# Third-order fit using Weighted Least Squares (WLS)
# Here weights are constant: 1/(2^2) for each observation.
weights <- rep(1 / (2^2), length(x))
model3_wls <- lm(y_tilde ~ poly(x, 3, raw = TRUE), weights = weights)
f3_wls <- predict(model3_wls, data.frame(x = x))
e3_wls <- sqrt(sum((y_tilde - f3_wls)^2))

# ------------------------------
# Display Fit Coefficients and Error Norms
# ------------------------------
cat("First-order fit coefficients:\n")
print(coef(model1))
cat(sprintf("First-order norm(error)  = %.4f\n\n", e1))

cat("Second-order fit coefficients:\n")
print(coef(model2))
cat(sprintf("Second-order norm(error) = %.4f\n\n", e2))

cat("Third-order fit coefficients:\n")
print(coef(model3))
cat(sprintf("Third-order norm(error)  = %.4f\n\n", e3))
cat(sprintf("Difference between LS and WLS (3rd-order) error: %.4f\n", e3 - e3_wls))

# ------------------------------
# Plotting the Fitted Models
# ------------------------------
par(mfrow = c(2,2))

# Plot raw data
plot(x, y_tilde, col = "blue", pch = 16, main = "Raw Data", xlab = "Time (day)", ylab = "Stock Value ($)")

# Plot 1st-order fit
plot(x, y_tilde, col = "blue", pch = 16, main = "1st-order Fit", xlab = "Time (day)", ylab = "Stock Value ($)")
lines(x, f1, col = "red", lwd = 1.5)

# Plot 1st and 2nd-order fits
plot(x, y_tilde, col = "blue", pch = 16, main = "1st & 2nd-order Fits", xlab = "Time (day)", ylab = "Stock Value ($)")
lines(x, f1, col = "red", lty = 2, lwd = 1.5)
lines(x, f2, col = "green", lwd = 1.5)
legend("topright", legend = c("1st-order", "2nd-order"), col = c("red", "green"), lty = c(2, 1))

# Plot 1st, 2nd and 3rd-order fits
plot(x, y_tilde, col = "blue", pch = 16, main = "1st, 2nd & 3rd-order Fits", xlab = "Time (day)", ylab = "Stock Value ($)")
lines(x, f1, col = "red", lty = 2, lwd = 1.5)
lines(x, f2, col = "green", lty = 3, lwd = 1.5)
lines(x, f3, col = "purple", lty = 4, lwd = 1.5)
legend("topright", legend = c("1st-order", "2nd-order", "3rd-order"),
       col = c("red", "green", "purple"), lty = c(2, 3, 4))
