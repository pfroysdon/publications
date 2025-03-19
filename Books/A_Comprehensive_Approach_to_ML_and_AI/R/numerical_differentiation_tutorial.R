# Numerical Differentiation Tutorial in R
#
# This tutorial demonstrates how to compute the numerical gradient of a scalar
# function using central differences. We define a simple function:
#    f(x) = sum(x^2)
# whose analytical gradient is 2*x.
#
# The function numericalGradient(f, x, h) computes the gradient at point x using:
#    grad[i] = (f(x + h*e_i) - f(x - h*e_i)) / (2*h)
# We compare the numerical gradient with the analytical gradient.

set.seed(1)

# Define the test function and its analytical gradient
f <- function(x) sum(x^2)
analyticalGradient <- function(x) 2 * x

# Choose a test point (5-dimensional random vector)
x0 <- rnorm(5)

# Function to compute numerical gradient using central differences
numericalGradient <- function(f, x, h = 1e-5) {
  grad <- numeric(length(x))
  for (i in seq_along(x)) {
    e <- rep(0, length(x))
    e[i] <- 1
    grad[i] <- (f(x + h * e) - f(x - h * e)) / (2 * h)
  }
  grad
}

numGrad <- numericalGradient(f, x0, h = 1e-5)

cat("Test point x0:\n"); print(x0)
cat("Analytical gradient:\n"); print(analyticalGradient(x0))
cat("Numerical gradient:\n"); print(numGrad)
cat(sprintf("Difference (L2 norm) between gradients: %.6f\n", norm(numGrad - analyticalGradient(x0), type = "2")))

# Plot comparison as a bar graph
barplot(rbind(analyticalGradient(x0), numGrad), beside = TRUE,
        col = c("blue", "red"), names.arg = 1:length(x0),
        main = "Comparison of Analytical and Numerical Gradients",
        xlab = "Dimension", ylab = "Gradient Value")
legend("topright", legend = c("Analytical", "Numerical"), fill = c("blue", "red"))
