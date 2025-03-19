# Sigmoid Test in R
#
# This script plots the sigmoid function over a range of x values,
# then shows how changing the weight and bias affects the function's shape.

# Define x range
x <- seq(-8, 8, by = 0.1)
sigmoid <- function(x) 1 / (1 + exp(-x))

# Plot basic sigmoid
f <- sigmoid(x)
plot(x, f, type = "l", main = "Sigmoid Function", xlab = "x", ylab = "f(x)")

# Adjusting the weights
w <- c(0.5, 1.0, 2.0)
colors <- c("red", "blue", "green")
plot(x, sigmoid(x * w[1]), type = "l", col = colors[1],
     ylim = c(0,1), main = "Effect of Weight Adjustment", xlab = "x", ylab = "h_w(x)")
for (i in 2:length(w)) {
  lines(x, sigmoid(x * w[i]), col = colors[i])
}
legend("bottomright", legend = paste("w =", w), col = colors, lty = 1)

# Effect of Bias
w_val <- 5.0
b <- c(-8.0, 0.0, 8.0)
colors <- c("red", "blue", "green")
plot(x, sigmoid(x * w_val + b[1]), type = "l", col = colors[1],
     ylim = c(0,1), main = "Effect of Bias Adjustment", xlab = "x", ylab = "h_wb(x)")
for (i in 2:length(b)) {
  lines(x, sigmoid(x * w_val + b[i]), col = colors[i])
}
legend("bottomright", legend = paste("b =", b), col = colors, lty = 1)
