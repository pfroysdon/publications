# Naive Bayes Tutorial in R
#
# This tutorial demonstrates a simple Naive Bayes classifier on 2D data.
# Synthetic data is generated for two classes, and a Gaussian Naive Bayes model
# is trained. The classifier is then used to predict a new point, and decision
# boundaries are visualized.

set.seed(1)

# -----------------------------
# Generate synthetic 2D data for two classes
# -----------------------------
X <- rbind(matrix(rnorm(50 * 2), ncol = 2),
           matrix(rnorm(50 * 2, mean = 0, sd = 1) + 3, ncol = 2))
y <- c(rep(1, 50), rep(2, 50))

# -----------------------------
# Naive Bayes Training Function
# -----------------------------
myNaiveBayes <- function(trainData, trainLabels) {
  classes <- sort(unique(trainLabels))
  numClasses <- length(classes)
  N <- nrow(trainData)
  d <- ncol(trainData)
  
  priors <- numeric(numClasses)
  means <- vector("list", numClasses)
  variances <- vector("list", numClasses)
  
  for (k in 1:numClasses) {
    idx <- which(trainLabels == classes[k])
    priors[k] <- length(idx) / N
    means[[k]] <- colMeans(trainData[idx, , drop = FALSE])
    variances[[k]] <- apply(trainData[idx, , drop = FALSE], 2, var)  # using 1/N normalization
  }
  
  list(classes = classes, priors = priors, means = means, variances = variances)
}

# -----------------------------
# Naive Bayes Classification Function
# -----------------------------
classifyNaiveBayes <- function(model, x) {
  numClasses <- length(model$classes)
  scores <- numeric(numClasses)
  
  for (k in 1:numClasses) {
    mu <- model$means[[k]]
    sigma2 <- model$variances[[k]]
    # Gaussian likelihood assuming independent features
    likelihood <- prod(1 / sqrt(2 * pi * sigma2) * exp(- ( (x - mu)^2 ) / (2 * sigma2)))
    scores[k] <- log(model$priors[k]) + log(likelihood)
  }
  model$classes[which.max(scores)]
}

# Train the Naive Bayes model
model <- myNaiveBayes(X, y)

# Classify a new point
newPoint <- c(0, 0)
predictedClass <- classifyNaiveBayes(model, newPoint)
cat("Predicted Class for new point [0,0]:", predictedClass, "\n")

# -----------------------------
# Visualize Decision Boundaries
# -----------------------------
x1range <- seq(min(X[,1]) - 1, max(X[,1]) + 1, length.out = 100)
x2range <- seq(min(X[,2]) - 1, max(X[,2]) + 1, length.out = 100)
grid <- expand.grid(x1 = x1range, x2 = x2range)

# Predict on grid points
predictions <- apply(grid, 1, function(pt) classifyNaiveBayes(model, as.numeric(pt)))
Z <- matrix(predictions, nrow = 100, ncol = 100, byrow = TRUE)

# Plot data and decision boundaries
par(mfrow = c(1,2))
plot(X[y==1,1], X[y==1,2], col = "red", pch = 16, main = "Naive Bayes Classification",
     xlab = "Feature 1", ylab = "Feature 2")
points(X[y==2,1], X[y==2,2], col = "blue", pch = 16)
legend("topright", legend = c("Class 1", "Class 2"), col = c("red", "blue"), pch = 16)

plot(X[y==1,1], X[y==1,2], col = "red", pch = 16, main = "Decision Boundaries",
     xlab = "Feature 1", ylab = "Feature 2")
points(X[y==2,1], X[y==2,2], col = "blue", pch = 16)
contour(x1range, x2range, Z, add = TRUE, lwd = 1.5)
legend("topright", legend = c("Class 1", "Class 2"), col = c("red", "blue"), pch = 16)
