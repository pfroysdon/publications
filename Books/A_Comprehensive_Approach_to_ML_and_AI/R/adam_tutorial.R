# adam_tutorial.R
# Adam Optimization Tutorial in R
# Minimizes f(x) = (x1 - 2)^2 + (x2 + 3)^2 using Adam.

set.seed(1)

# Define objective function and gradient
f <- function(x) { (x[1] - 2)^2 + (x[2] + 3)^2 }
grad_f <- function(x) { c(2*(x[1] - 2), 2*(x[2] + 3)) }

# Adam parameters
alpha <- 0.1
beta1 <- 0.9
beta2 <- 0.999
epsilon <- 1e-8
numIterations <- 1000

# Initialization
x <- c(-5, 5)
m <- c(0, 0)
v <- c(0, 0)
lossHistory <- numeric(numIterations)

for (t in 1:numIterations) {
  g <- grad_f(x)
  m <- beta1 * m + (1 - beta1) * g
  v <- beta2 * v + (1 - beta2) * (g^2)
  m_hat <- m / (1 - beta1^t)
  v_hat <- v / (1 - beta2^t)
  x <- x - alpha * m_hat / (sqrt(v_hat) + epsilon)
  lossHistory[t] <- f(x)
  if (t %% 100 == 0) {
    cat(sprintf("Iteration %d: Loss = %.4f, x = [%.4f, %.4f]\n", t, lossHistory[t], x[1], x[2]))
  }
}

cat(sprintf("Optimized solution: x = [%.4f, %.4f]\n", x[1], x[2]))
cat(sprintf("Final objective value: %.4f\n", f(x)))

plot(1:numIterations, lossHistory, type = "l", lwd = 2,
     xlab = "Iteration", ylab = "Objective Function Value", main = "Adam Optimization Convergence")
grid()
