# accuracy_vs_precision_tutorial.R
# This tutorial visualizes accuracy versus precision concepts by plotting
# concentric circles (with markers) and corresponding example distributions.
# Each subplot illustrates a different configuration.

# Clear workspace and close graphics devices
rm(list = ls())
graphics.off()

# Set up common parameters
radii <- c(1, 2, 3, 4, 5)      # Radii of concentric circles
nCirclePoints <- 200          # Number of points to plot each circle

# Define a helper function to compute circle coordinates
circle <- function(x0, y0, r, N = 200) {
  theta <- seq(0, 2 * pi, length.out = N)
  xC <- x0 + r * cos(theta)
  yC <- y0 + r * sin(theta)
  list(x = xC, y = yC)
}

# Set up a 2x4 grid for subplots
par(mfrow = c(2, 4), mar = c(4, 4, 2, 1))

# ---- Subplot (a): Top Row ----
# Plot concentric circles centered at (0,0) with several 'X' markers.
plot(NA, xlim = c(-5, 5), ylim = c(-5, 5), xlab = "X", ylab = "Y",
     main = "(a)", asp = 1)
for (r in radii) {
  circ <- circle(0, 0, r, nCirclePoints)
  lines(circ$x, circ$y, col = "black")
}
# Manually place 'X' markers at specified angles and radii
points(1.5 * cos(0 * pi/180), 1.5 * sin(0 * pi/180), pch = 4, col = "black", cex = 2, lwd = 2)
points(4.5 * cos(80 * pi/180), 4.5 * sin(80 * pi/180), pch = 4, col = "black", cex = 2, lwd = 2)
points(4.5 * cos(150 * pi/180), 4.5 * sin(150 * pi/180), pch = 4, col = "black", cex = 2, lwd = 2)
points(2.5 * cos(200 * pi/180), 2.5 * sin(200 * pi/180), pch = 4, col = "black", cex = 2, lwd = 2)
points(4.5 * cos(300 * pi/180), 4.5 * sin(300 * pi/180), pch = 4, col = "black", cex = 2, lwd = 2)
# Add dashed vertical lines
abline(v = 0, col = "red", lty = 2, lwd = 1.5)
abline(v = 0.75, col = "green", lty = 2, lwd = 1.5)
box()

# ---- Subplot (a): Bottom Row ----
# Plot an example distribution (normal PDF with mean=0.75, sd=2)
subplot_a <- function() {
  xvals <- seq(-10, 10, length.out = 200)
  mu_a <- 0.75
  sigma_a <- 2
  yvals_a <- dnorm(xvals, mean = mu_a, sd = sigma_a)
  plot(xvals, yvals_a, type = "l", lwd = 1.5, col = "black",
       xlab = "x", ylab = "PDF", main = "(a) Distribution", xlim = c(-5, 5), asp = 1)
  abline(v = 0, col = "red", lty = 2, lwd = 1.5)
  abline(v = mu_a, col = "green", lty = 2, lwd = 1.5)
  grid()
}
plot.new()
par(mfrow = c(2, 4))
subplot_a()  # Will be drawn in the 5th subplot

# ---- Subplot (b): Top Row ----
# Plot circles (again) with 'X' markers at a circle of radius 2 at angles 0, 90, 180, 270, and at center.
plot(NA, xlim = c(-5, 5), ylim = c(-5, 5), xlab = "X", ylab = "Y", main = "(b)", asp = 1)
for (r in radii) {
  circ <- circle(0, 0, r, nCirclePoints)
  lines(circ$x, circ$y, col = "black")
}
points(2 * cos(0 * pi/180), 2 * sin(0 * pi/180), pch = 4, col = "black", cex = 2, lwd = 2)
points(2 * cos(90 * pi/180), 2 * sin(90 * pi/180), pch = 4, col = "black", cex = 2, lwd = 2)
points(2 * cos(180 * pi/180), 2 * sin(180 * pi/180), pch = 4, col = "black", cex = 2, lwd = 2)
points(2 * cos(270 * pi/180), 2 * sin(270 * pi/180), pch = 4, col = "black", cex = 2, lwd = 2)
points(0, 0, pch = 4, col = "black", cex = 2, lwd = 2)
abline(v = 0, col = "red", lty = 2, lwd = 1.5)
abline(v = 0, col = "green", lty = 2, lwd = 1.5)
box()

# ---- Subplot (b): Bottom Row ----
# Plot normal PDF with mean 0, sd 1
plot(xvals, dnorm(xvals, mean = 0, sd = 1), type = "l", lwd = 1.5, col = "black",
     xlab = "x", ylab = "PDF", main = "(b) Distribution", xlim = c(-5, 5), asp = 1)
abline(v = 0, col = "red", lty = 2, lwd = 1.5)
abline(v = 0, col = "green", lty = 2, lwd = 1.5)
grid()

# ---- Subplot (c): Top Row ----
# Plot circles with 'X' markers at angles between 100 and 108 degrees on circle of radius 4.5.
plot(NA, xlim = c(-5, 5), ylim = c(-5, 5), xlab = "X", ylab = "Y", main = "(c)", asp = 1)
for (r in radii) {
  circ <- circle(0, 0, r, nCirclePoints)
  lines(circ$x, circ$y, col = "black")
}
for (theta in seq(100, 108, by = 2)) {
  points(4.5 * cos(theta * pi/180), 4.5 * sin(theta * pi/180), pch = 4, col = "black", cex = 2, lwd = 2)
}
abline(v = 0, col = "red", lty = 2, lwd = 1.5)
abline(v = -1.2, col = "green", lty = 2, lwd = 1.5)
box()

# ---- Subplot (c): Bottom Row ----
# Plot normal PDF with mean -1.2 and sd 0.2
plot(xvals, dnorm(xvals, mean = -1.2, sd = 0.2), type = "l", lwd = 1.5, col = "black",
     xlab = "x", ylab = "PDF", main = "(c) Distribution", xlim = c(-5, 5), asp = 1)
abline(v = 0, col = "red", lty = 2, lwd = 1.5)
abline(v = -1.2, col = "green", lty = 2, lwd = 1.5)
grid()

# ---- Subplot (d): Top Row ----
# Plot circles with small 'X' markers on a circle of radius 0.5
plot(NA, xlim = c(-5, 5), ylim = c(-5, 5), xlab = "X", ylab = "Y", main = "(d)", asp = 1)
for (r in radii) {
  circ <- circle(0, 0, r, nCirclePoints)
  lines(circ$x, circ$y, col = "black")
}
points(0.5 * cos(0 * pi/180), 0.5 * sin(0 * pi/180), pch = 4, col = "black", cex = 2, lwd = 2)
points(0.5 * cos(90 * pi/180), 0.5 * sin(90 * pi/180), pch = 4, col = "black", cex = 2, lwd = 2)
points(0.5 * cos(180 * pi/180), 0.5 * sin(180 * pi/180), pch = 4, col = "black", cex = 2, lwd = 2)
points(0.5 * cos(270 * pi/180), 0.5 * sin(270 * pi/180), pch = 4, col = "black", cex = 2, lwd = 2)
points(0, 0, pch = 4, col = "black", cex = 2, lwd = 2)
abline(v = 0, col = "red", lty = 2, lwd = 1.5)
abline(v = 0, col = "green", lty = 2, lwd = 1.5)
box()

# ---- Subplot (d): Bottom Row ----
# Plot normal PDF with mean 0 and sd 0.2
plot(xvals, dnorm(xvals, mean = 0, sd = 0.2), type = "l", lwd = 1.5, col = "black",
     xlab = "x", ylab = "PDF", main = "(d) Distribution", xlim = c(-5, 5), asp = 1)
abline(v = 0, col = "red", lty = 2, lwd = 1.5)
abline(v = 0, col = "green", lty = 2, lwd = 1.5)
grid()

# (Optional) To save the figure, use the png() function:
# png("accuracy_vs_precision.png", width = 1200, height = 600)
# ... [plotting code] ...
# dev.off()
