# Clear workspace and close graphics
rm(list = ls())
graphics.off()

# ------------------------------
# First Part: Mixture of Gaussians KDE
# ------------------------------

# Define data (SixMPG values)
SixMPG <- c(13, 15, 23, 29, 32, 34)

# Plot histogram of the data
hist(SixMPG, main = "Histogram of SixMPG", xlab = "SixMPG")

# Use kernel density estimation with a fixed bandwidth (bw = 4)
# Note: density() in R will estimate over a grid; we force the grid from 0 to 45
x <- seq(0, 45, by = 0.1)
pdSix <- density(SixMPG, bw = 4, from = 0, to = 45, n = length(x))

# Plot the KDE estimate (in black, with line width 2)
plot(pdSix$x, pdSix$y, type = "l", col = "black", lwd = 2,
     xlab = "X", ylab = "Estimated Density",
     main = "Kernel Density Estimate for Mixture of Gaussians")

# Plot individual Gaussian PDFs scaled by 1/6
# For each data point, create a Gaussian with mean = value and sd = 4.
for (i in seq_along(SixMPG)) {
  y_indiv <- dnorm(x, mean = SixMPG[i], sd = 4) / 6
  lines(x, y_indiv, col = "blue", lty = 3, lwd = 2)  # lty=3 gives a dotted line
}

# ------------------------------
# Second Part: Automotive MPG KDE
# ------------------------------

# Load or simulate automotive MPG data.
# (If you have a CSV or data file, replace the simulation with a load command.)
set.seed(123)
MPG <- rnorm(100, mean = 30, sd = 5)

# Define grid for evaluation
x2 <- seq(-10, 60, by = 1)

# Compute kernel density estimates with different bandwidths.
pd1 <- density(MPG, kernel = "gaussian")         # default bandwidth
pd2 <- density(MPG, kernel = "gaussian", bw = 1)     # bandwidth = 1
pd3 <- density(MPG, kernel = "gaussian", bw = 5)     # bandwidth = 5

# Interpolate densities at the grid x2
y1 <- approx(pd1$x, pd1$y, xout = x2)$y
y2 <- approx(pd2$x, pd2$y, xout = x2)$y
y3 <- approx(pd3$x, pd3$y, xout = x2)$y

# Plot the three density estimates on the same figure.
plot(x2, y1, type = "l", col = "red", lwd = 2,
     xlab = "X", ylab = "Estimated Density",
     main = "Kernel Density Estimate for Automotive MPG")
lines(x2, y2, col = "black", lty = 3, lwd = 2)
lines(x2, y3, col = "blue", lty = 2, lwd = 2)

legend("topright", legend = c("Bandwidth = Default", "Bandwidth = 1", "Bandwidth = 5"),
       col = c("red", "black", "blue"), lty = c(1, 3, 2), lwd = 2)
