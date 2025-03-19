# dbscan_tutorial.R
# This script implements DBSCAN from scratch on synthetic 2D data and visualizes the resulting clusters.

rm(list=ls())
graphics.off()
set.seed(1)

# Generate synthetic 2D data
X <- rbind(matrix(rnorm(50 * 2), ncol = 2), matrix(rnorm(50 * 2) + 3, ncol = 2))
y <- c(rep(1, 50), rep(2, 50))

# DBSCAN parameters
eps <- 0.8
minPts <- 5

# DBSCAN clustering function
dbscanClustering <- function(X, eps, minPts) {
  n <- nrow(X)
  labels <- rep(0, n)    # 0 means not yet assigned
  visited <- rep(FALSE, n)
  clusterId <- 0
  
  regionQuery <- function(X, idx, eps) {
    point <- X[idx, ]
    distances <- sqrt(rowSums((X - matrix(rep(point, n), nrow = n, byrow = TRUE))^2))
    which(distances <= eps)
  }
  
  for (i in 1:n) {
    if (!visited[i]) {
      visited[i] <- TRUE
      Neighbors <- regionQuery(X, i, eps)
      if (length(Neighbors) < minPts) {
        labels[i] <- -1
      } else {
        clusterId <- clusterId + 1
        labels[i] <- clusterId
        seedSet <- Neighbors
        k <- 1
        while (k <= length(seedSet)) {
          j <- seedSet[k]
          if (!visited[j]) {
            visited[j] <- TRUE
            Neighbors_j <- regionQuery(X, j, eps)
            if (length(Neighbors_j) >= minPts) {
              seedSet <- unique(c(seedSet, Neighbors_j))
            }
          }
          if (labels[j] == 0) {
            labels[j] <- clusterId
          }
          k <- k + 1
        }
      }
    }
  }
  labels
}

labels <- dbscanClustering(X, eps, minPts)

# Visualize clustering results
par(mfrow = c(1, 2))
plot(X, col = "blue", pch = 16, main = "DBSCAN Clustering")
text(X, labels, cex = 0.8)
plot(X, col = ifelse(y == 1, "red", "blue"), pch = 16,
     main = sprintf("DBSCAN Clustering (eps = %.2f, minPts = %d)", eps, minPts))
points(X, col = labels + 2, pch = 1)
legend("topright", legend = c("Class 1", "Class 2"), col = c("red", "blue"), pch = 16)
cat(sprintf("Number of clusters found: %d\n", length(unique(labels[labels > 0])))
)
cat(sprintf("Number of noise points: %d\n", sum(labels == -1)))
