# Clear workspace and close figures
rm(list = ls())
graphics.off()

set.seed(1)
# ------------------------------
# Generate Synthetic Data for Two Classes
# ------------------------------
N <- 100
X1 <- matrix(rnorm(N * 2), ncol = 2) + 2
X2 <- matrix(rnorm(N * 2), ncol = 2) - 2
X <- rbind(X1, X2)
y <- c(rep(1, N), rep(2, N))  # Class labels: 1 and 2

# ------------------------------
# Define LDA Function (Manual Implementation)
# ------------------------------
lda_manual <- function(X, y) {
  n <- nrow(X)
  d <- ncol(X)
  classes <- unique(y)
  C <- length(classes)
  
  # Compute overall mean of data
  mu <- colMeans(X)
  
  # Initialize within-class (Sw) and between-class (Sb) scatter matrices
  Sw <- matrix(0, nrow = d, ncol = d)
  Sb <- matrix(0, nrow = d, ncol = d)
  
  for (i in 1:C) {
    Xi <- X[y == classes[i], , drop = FALSE]
    Ni <- nrow(Xi)
    mu_i <- colMeans(Xi)
    
    # Within-class scatter
    Sw <- Sw + t(scale(Xi, center = mu_i, scale = FALSE)) %*% scale(Xi, center = mu_i, scale = FALSE)
    
    # Between-class scatter
    diff <- matrix(mu_i - mu, ncol = 1)
    Sb <- Sb + Ni * (diff %*% t(diff))
  }
  
  # Solve the generalized eigenvalue problem: Sb * v = lambda * Sw * v
  eig <- eigen(solve(Sw) %*% Sb)
  order_idx <- order(eig$values, decreasing = TRUE)
  W <- eig$vectors[, order_idx, drop = FALSE]
  
  # Project the data onto the new subspace
  projectedData <- X %*% W
  return(list(W = W, projectedData = projectedData))
}

# Run LDA
lda_result <- lda_manual(X, y)
W <- lda_result$W
projectedData <- lda_result$projectedData

# ------------------------------
# Plot the Original and Projected Data
# ------------------------------
par(mfrow = c(1, 2))

# Subplot 1: Original Data with Different Colors for Classes
plot(X1, col = "red", pch = 16, xlab = "Feature 1", ylab = "Feature 2", main = "Original Data")
points(X2, col = "blue", pch = 16)
grid()

# Draw a dashed line representing the first LDA direction
mu_all <- colMeans(X)
dirVec <- W[,1]  # Primary discriminant direction
t_vals <- seq(-80, 80, length.out = 100)
linePoints <- t(sapply(t_vals, function(t) mu_all + t * dirVec))
lines(linePoints, col = "black", lty = 2, lwd = 2)

# Subplot 2: Data Projected onto the First Discriminant Axis
proj_class1 <- projectedData[1:N, 1]
proj_class2 <- projectedData[(N + 1):(2 * N), 1]
plot(proj_class1, rep(0, length(proj_class1)), col = "red", pch = 16,
     xlab = "Projection Value", ylab = "", main = "Data Projected onto First Discriminant",
     xlim = range(projectedData[,1]))
points(proj_class2, rep(0, length(proj_class2)), col = "blue", pch = 16)
grid()
