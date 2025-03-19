# adaboost_tutorial.R
# AdaBoost Tutorial in R using decision stumps.
# This script generates synthetic data, trains AdaBoost, and plots the decision boundaries.

set.seed(1)

# Generate synthetic data
N <- 100
X1 <- matrix(rnorm(N * 2), ncol = 2) + 2    # Class +1
X2 <- matrix(rnorm(N * 2), ncol = 2) - 2    # Class -1
X <- rbind(X1, X2)
y <- c(rep(1, N), rep(-1, N))

# AdaBoost training functions
decisionStump <- function(X, y, weights) {
  n <- nrow(X); d <- ncol(X)
  best_error <- Inf
  best_stump <- list(feature = NA, threshold = NA, polarity = NA)
  best_pred <- rep(0, n)
  
  for (j in 1:d) {
    for (thresh in unique(X[, j])) {
      for (polarity in c(1, -1)) {
        pred <- rep(1, n)
        if (polarity == 1) {
          pred[X[, j] >= thresh] <- -1
        } else {
          pred[X[, j] >= thresh] <- 1
        }
        error <- sum(weights * (pred != y))
        if (error < best_error) {
          best_error <- error
          best_stump$feature <- j
          best_stump$threshold <- thresh
          best_stump$polarity <- polarity
          best_pred <- pred
        }
      }
    }
  }
  list(stump = best_stump, error = best_error, pred = best_pred)
}

adaboostTrain <- function(X, y, T) {
  n <- nrow(X)
  weights <- rep(1/n, n)
  stumps <- vector("list", T)
  alphas <- numeric(T)
  
  for (t in 1:T) {
    stump_info <- decisionStump(X, y, weights)
    stump <- stump_info$stump
    error <- max(stump_info$error, 1e-10)
    alpha <- 0.5 * log((1 - error) / error)
    stumps[[t]] <- stump
    alphas[t] <- alpha
    weights <- weights * exp(-alpha * y * stump_info$pred)
    weights <- weights / sum(weights)
  }
  list(stumps = stumps, alphas = alphas)
}

decisionStumpPredict <- function(stump, X) {
  n <- nrow(X)
  pred <- rep(1, n)
  if (stump$polarity == 1) {
    pred[X[, stump$feature] >= stump$threshold] <- -1
  } else {
    pred[X[, stump$feature] >= stump$threshold] <- 1
  }
  pred
}

adaboostPredict <- function(X, models) {
  T <- length(models$stumps)
  n <- nrow(X)
  agg <- rep(0, n)
  for (t in 1:T) {
    pred <- decisionStumpPredict(models$stumps[[t]], X)
    agg <- agg + models$alphas[t] * pred
  }
  preds <- sign(agg)
  preds[preds == 0] <- 1
  preds
}

# Train AdaBoost
T_rounds <- 50
boosting_models <- adaboostTrain(X, y, T_rounds)
y_pred_boosting <- adaboostPredict(X, boosting_models)
acc_boosting <- mean(y_pred_boosting == y) * 100
cat(sprintf("Boosting Training Accuracy: %.2f%%\n", acc_boosting))

# Plot decision boundary
x1_range <- seq(min(X[,1]) - 1, max(X[,1]) + 1, length.out = 100)
x2_range <- seq(min(X[,2]) - 1, max(X[,2]) + 1, length.out = 100)
grid_points <- expand.grid(x1 = x1_range, x2 = x2_range)
preds <- adaboostPredict(as.matrix(grid_points), boosting_models)
preds_mat <- matrix(preds, nrow = 100, ncol = 100)

par(mfrow = c(1,2))
# Plot original data and decision boundary
plot(X[y==1,1], X[y==1,2], col = "blue", pch = 16, xlab = "Feature 1", ylab = "Feature 2",
     main = "AdaBoost Classification (Clusters)")
points(X[y==-1,1], X[y==-1,2], col = "red", pch = 16)
contour(x1_range, x2_range, preds_mat, levels = 0, add = TRUE, lwd = 2, col = "black")
legend("topright", legend = c("Class +1", "Class -1"), col = c("blue", "red"), pch = 16)
