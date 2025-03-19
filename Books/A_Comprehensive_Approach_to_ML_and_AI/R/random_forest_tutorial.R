# Random Forest Tutorial in R
#
# This tutorial generates synthetic 2D data for two classes, trains a random forest
# model (using decision trees via rpart), predicts labels on training data, computes accuracy,
# and visualizes the decision boundary.

set.seed(42)
library(rpart)
library(rpart.plot)

# Generate synthetic data
num_samples <- 200
# Class 1: centered at (2,2)
X1 <- matrix(rnorm(num_samples/2 * 2), ncol = 2) + 2
Y1 <- rep(0, num_samples/2)
# Class 2: centered at (-2,-2)
X2 <- matrix(rnorm(num_samples/2 * 2), ncol = 2) - 2
Y2 <- rep(1, num_samples/2)
X <- rbind(X1, X2)
Y <- factor(c(Y1, Y2))

# Shuffle data
idx <- sample(1:num_samples)
X <- X[idx, ]
Y <- Y[idx]

# Plot dataset
plot(X[Y==0,1], X[Y==0,2], col = "red", pch = 16, xlab = "Feature 1", ylab = "Feature 2", main = "Random Forest Classification")
points(X[Y==1,1], X[Y==1,2], col = "blue", pch = 16)
legend("topright", legend = c("Class 0", "Class 1"), col = c("red", "blue"), pch = 16)
grid()

# Train Random Forest model using rpart (simulate ensemble by training multiple trees)
num_trees <- 20
max_depth <- 4
trees <- vector("list", num_trees)
for (t in 1:num_trees) {
  boot_idx <- sample(1:nrow(X), replace = TRUE)
  trees[[t]] <- rpart(Y ~ ., data = data.frame(X, Y), control = rpart.control(maxdepth = max_depth, minsplit = 5))
}

# Prediction function: majority vote from trees
predict_forest <- function(trees, newdata) {
  preds <- sapply(trees, function(tree) as.numeric(as.character(predict(tree, newdata = newdata, type = "class"))))
  apply(preds, 1, function(x) as.numeric(names(which.max(table(x)))))
}

# Predict on training data
Y_pred <- predict_forest(trees, data.frame(X))
accuracy <- mean(Y_pred == as.numeric(as.character(Y))) * 100
cat(sprintf("Model Accuracy: %.2f%%\n", accuracy))

# Create a mesh grid for visualization
x1_range <- seq(min(X[,1]) - 1, max(X[,1]) + 1, length.out = 100)
x2_range <- seq(min(X[,2]) - 1, max(X[,2]) + 1, length.out = 100)
grid <- expand.grid(x1 = x1_range, x2 = x2_range)
grid_preds <- predict_forest(trees, grid)
grid_pred_matrix <- matrix(grid_preds, nrow = 100, ncol = 100)

# Plot decision boundary
library(ggplot2)
df_grid <- data.frame(expand.grid(x1 = x1_range, x2 = x2_range), pred = as.factor(as.vector(grid_pred_matrix)))
ggplot(df_grid, aes(x = x1, y = x2, fill = pred)) +
  geom_tile(alpha = 0.3) +
  geom_point(data = data.frame(X, Y), aes(x = X1, y = X2, color = Y), size = 2) +
  labs(title = "Random Forest Decision Boundary", x = "Feature 1", y = "Feature 2") +
  scale_fill_manual(values = c("red", "blue")) +
  scale_color_manual(values = c("red", "blue")) +
  theme_minimal()
