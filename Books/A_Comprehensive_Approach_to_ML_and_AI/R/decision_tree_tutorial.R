# decision_tree_tutorial.R
# This script demonstrates a decision tree classifier built from scratch using the Gini impurity criterion.
# It generates a 2D dataset, recursively builds a tree, and visualizes the decision boundary.

rm(list=ls())
graphics.off()
set.seed(1)

# Generate synthetic data
N <- 100
X1 <- matrix(rnorm(N * 2), ncol = 2) + 2
X2 <- matrix(rnorm(N * 2), ncol = 2) - 2
X <- rbind(X1, X2)
y <- c(rep(1, N), rep(-1, N))

# Functions to build the decision tree
majorityVote <- function(y) {
  if (sum(y == 1) >= sum(y == -1)) 1 else -1
}

giniImpurity <- function(y) {
  n <- length(y)
  if (n == 0) return(0)
  p <- sum(y == 1) / n
  1 - (p^2 + (1 - p)^2)
}

findBestSplit <- function(X, y) {
  n <- nrow(X)
  bestImpurity <- Inf
  bestFeature <- NULL
  bestThreshold <- NULL
  for (j in 1:ncol(X)) {
    values <- sort(unique(X[, j]))
    for (thresh in values) {
      split <- X[, j] < thresh
      if (sum(split) == 0 || sum(!split) == 0) next
      impurityLeft <- giniImpurity(y[split])
      impurityRight <- giniImpurity(y[!split])
      weightedImpurity <- (sum(split) / n) * impurityLeft + (sum(!split) / n) * impurityRight
      if (weightedImpurity < bestImpurity) {
        bestImpurity <- weightedImpurity
        bestFeature <- j
        bestThreshold <- thresh
      }
    }
  }
  list(feature = bestFeature, threshold = bestThreshold)
}

buildTree <- function(X, y, depth, maxDepth) {
  if (all(y == y[1]) || depth >= maxDepth || nrow(X) < 2) {
    return(list(isLeaf = TRUE, prediction = majorityVote(y)))
  }
  split <- findBestSplit(X, y)
  if (is.null(split$feature)) {
    return(list(isLeaf = TRUE, prediction = majorityVote(y)))
  }
  left_indices <- X[, split$feature] < split$threshold
  if (sum(left_indices) == 0 || sum(!left_indices) == 0) {
    return(list(isLeaf = TRUE, prediction = majorityVote(y)))
  }
  left_tree <- buildTree(X[left_indices, , drop = FALSE], y[left_indices], depth + 1, maxDepth)
  right_tree <- buildTree(X[!left_indices, , drop = FALSE], y[!left_indices], depth + 1, maxDepth)
  list(isLeaf = FALSE, feature = split$feature, threshold = split$threshold, left = left_tree, right = right_tree)
}

predictTree <- function(tree, X) {
  preds <- numeric(nrow(X))
  for (i in 1:nrow(X)) {
    preds[i] <- traverseTree(tree, X[i, ])
  }
  preds
}

traverseTree <- function(tree, x) {
  if (tree$isLeaf) return(tree$prediction)
  if (x[tree$feature] < tree$threshold)
    traverseTree(tree$left, x)
  else
    traverseTree(tree$right, x)
}

tree <- buildTree(X, y, depth = 0, maxDepth = 3)

# Create a grid for decision boundary visualization
x_min <- min(X[, 1]) - 1; x_max <- max(X[, 1]) + 1
y_min <- min(X[, 2]) - 1; y_max <- max(X[, 2]) + 1
xGrid <- seq(x_min, x_max, length.out = 200)
yGrid <- seq(y_min, y_max, length.out = 200)
grid <- expand.grid(x1 = xGrid, x2 = yGrid)
preds <- predictTree(tree, as.matrix(grid))
preds_matrix <- matrix(preds, nrow = length(xGrid), ncol = length(yGrid))

# Plot data and decision boundaries
par(mfrow = c(1, 2))
plot(X[y == 1, 1], X[y == 1, 2], col = "blue", pch = 16, xlab = "Feature 1", ylab = "Feature 2",
     main = "Decision Tree Classification")
points(X[y == -1, 1], X[y == -1, 2], col = "red", pch = 16)
contour(xGrid, yGrid, preds_matrix, levels = 0, add = TRUE, lty = 2, lwd = 2)
legend("topright", legend = c("Class +1", "Class -1"), col = c("blue", "red"), pch = 16)

plot(X[y == 1, 1], X[y == 1, 2], col = "blue", pch = 16, xlab = "Feature 1", ylab = "Feature 2",
     main = "Decision Tree with Decision Boundary")
points(X[y == -1, 1], X[y == -1, 2], col = "red", pch = 16)
filled.contour(xGrid, yGrid, preds_matrix, color.palette = terrain.colors,
               plot.title = title("Decision Boundary"), plot.axes = {axis(1); axis(2); points(X, col = ifelse(y == 1, "blue", "red"), pch = 16)})
