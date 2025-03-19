# bootstrap_bagging_boosting_tutorial.R
# This demo illustrates bootstrap, bagging, and boosting using decision stumps.
# It generates synthetic 2D data for two classes, trains bagging and AdaBoost ensembles,
# and plots the decision boundaries.

set.seed(1)

# Generate synthetic data
N <- 200
X_class1 <- matrix(rnorm(100 * 2), ncol = 2) + c(-1, -1)
X_class2 <- matrix(rnorm(100 * 2), ncol = 2) + c(1, 1)
X <- rbind(X_class1, X_class2)
y <- c(rep(-1, 100), rep(1, 100))

# Define decision stump functions
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
        err <- sum(weights * (pred != y))
        if (err < best_error) {
          best_error <- err
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

# Bagging ensemble functions
baggingEnsembleTrain <- function(X, y, T) {
  n <- nrow(X)
  models <- vector("list", T)
  for (t in 1:T) {
    idx <- sample(1:n, n, replace = TRUE)
    X_boot <- X[idx, ]
    y_boot <- y[idx]
    weights <- rep(1/n, n)
    stump_info <- decisionStump(X_boot, y_boot, weights)
    models[[t]] <- stump_info$stump
  }
  models
}

baggingEnsemblePredict <- function(models, X) {
  T <- length(models)
  n <- nrow(X)
  pred_matrix <- matrix(0, n, T)
  for (t in 1:T) {
    stump <- models[[t]]
    pred <- rep(1, n)
    if (stump$polarity == 1) {
      pred[X[, stump$feature] >= stump$threshold] <- -1
    } else {
      pred[X[, stump$feature] >= stump$threshold] <- 1
    }
    pred_matrix[, t] <- pred
  }
  predictions <- sign(rowSums(pred_matrix))
  predictions[predictions == 0] <- 1
  predictions
}

# Boosting ensemble functions (AdaBoost)
adaboostTrain <- function(X, y, T) {
  n <- nrow(X)
  weights <- rep(1/n, n)
  models <- vector("list", T)
  alphas <- numeric(T)
  for (t in 1:T) {
    stump_info <- decisionStump(X, y, weights)
    stump <- stump_info$stump
    error <- max(stump_info$error, 1e-10)
    alpha <- 0.5 * log((1 - error) / error)
    models[[t]] <- stump
    alphas[t] <- alpha
    weights <- weights * exp(-alpha * y * stump_info$pred)
    weights <- weights / sum(weights)
  }
  list(stumps = models, alphas = alphas)
}

adaboostPredict <- function(X, models) {
  T <- length(models$stumps)
  n <- nrow(X)
  agg <- rep(0, n)
  for (t in 1:T) {
    stump <- models$stumps[[t]]
    pred <- rep(1, n)
    if (stump$polarity == 1) {
      pred[X[, stump$feature] >= stump$threshold] <- -1
    } else {
      pred[X[, stump$feature] >= stump$threshold] <- 1
    }
    agg <- agg + models$alphas[t] * pred
  }
  preds <- sign(agg)
  preds[preds == 0] <- 1
  preds
}

# Train ensembles
T_rounds <- 50
bagging_models <- baggingEnsembleTrain(X, y, T_rounds)
boosting_models <- adaboostTrain(X, y, T_rounds)

y_pred_bagging <- baggingEnsemblePredict(bagging_models, X)
acc_bagging <- mean(y_pred_bagging == y) * 100
cat(sprintf("Bagging Training Accuracy: %.2f%%\n", acc_bagging))

y_pred_boosting <- adaboostPredict(X, boosting_models)
acc_boosting <- mean(y_pred_boosting == y) * 100
cat(sprintf("Boosting Training Accuracy: %.2f%%\n", acc_boosting))

# Plot decision boundaries
x1_range <- seq(min(X[,1]) - 1, max(X[,1]) + 1, length.out = 100)
x2_range <- seq(min(X[,2]) - 1, max(X[,2]) + 1, length.out = 100)
grid_points <- expand.grid(x1 = x1_range, x2 = x2_range)

preds_bagging <- baggingEnsemblePredict(bagging_models, as.matrix(grid_points))
preds_boosting <- adaboostPredict(as.matrix(grid_points), boosting_models)

preds_bagging_mat <- matrix(preds_bagging, nrow = 100, ncol = 100)
preds_boosting_mat <- matrix(preds_boosting, nrow = 100, ncol = 100)

par(mfrow = c(1,2))
image(x1_range, x2_range, preds_bagging_mat, col = c("red", "blue"),
      main = sprintf("Bagging Decision Boundary (T = %d)", T_rounds),
      xlab = "x1", ylab = "x2")
points(X[y==-1,1], X[y==-1,2], col = "red", pch = 16)
points(X[y==1,1], X[y==1,2], col = "blue", pch = 16)

image(x1_range, x2_range, preds_boosting_mat, col = c("red", "blue"),
      main = sprintf("Boosting Decision Boundary (T = %d)", T_rounds),
      xlab = "x1", ylab = "x2")
points(X[y==-1,1], X[y==-1,2], col = "red", pch = 16)
points(X[y==1,1], X[y==1,2], col = "blue", pch = 16)
}

bootstrap_vs_bagging_vs_boosting_demo()
