# feature_engineering_tutorial.R
# Feature Engineering Pipeline Tutorial
#
# This tutorial demonstrates a complete feature engineering pipeline for a
# binary classification problem. The pipeline includes:
#
#   1. Data Cleaning: Handling missing values (via mean imputation) and
#      removing anomalies (samples with feature values beyond a threshold).
#
#   2. Feature Transformation: Normalizing the data using z-score scaling.
#
#   3. Feature Extraction: Applying PCA to extract principal components.
#
#   4. Feature Derivation: Creating polynomial features (degree 2) by adding
#      squared terms and pairwise products.
#
# The synthetic dataset is generated from two Gaussian distributions in 2D,
# representing two classes. Missing values and anomalies are introduced
# intentionally to demonstrate the cleaning process.

rm(list = ls())
graphics.off()
set.seed(1)

## 1. Generate Synthetic Dataset
N <- 200  # total number of samples
# Class 0: centered at (1,1)
X_class0 <- matrix(rnorm(2 * (N/2), mean = 1, sd = 0.5), nrow = 2)
# Class 1: centered at (3,3)
X_class1 <- matrix(rnorm(2 * (N/2), mean = 3, sd = 0.5), nrow = 2)
X <- cbind(X_class0, X_class1)  # 2 x N matrix
y <- c(rep(0, N/2), rep(1, N/2))  # binary labels

# Introduce missing values: randomly set 5% of entries in X to NA.
numMissing <- round(0.05 * length(X))
missingIndices <- sample(1:length(X), numMissing)
X[missingIndices] <- NA

# Introduce anomalies: randomly select 5 samples and multiply feature 1 by 10.
anomalyIndices <- sample(1:N, 5)
X[1, anomalyIndices] <- X[1, anomalyIndices] * 10

## 2. Data Cleaning
imputeMissing <- function(X) {
  X_imputed <- X
  for (j in 1:nrow(X)) {
    col <- X[j, ]
    if (any(is.na(col))) {
      col[is.na(col)] <- mean(col, na.rm = TRUE)
      X_imputed[j, ] <- col
    }
  }
  X_imputed
}
X_clean <- imputeMissing(X)

removeOutliers <- function(X, y, threshold = 2) {
  # X: features (d x N), y: vector of labels
  mu <- apply(X, 1, mean)
  sigma <- apply(X, 1, sd)
  keep <- rep(TRUE, ncol(X))
  for (i in 1:ncol(X)) {
    sample <- X[, i]
    if (any(abs((sample - mu) / sigma) >= threshold)) {
      keep[i] <- FALSE
    }
  }
  list(X_clean = X[, keep, drop = FALSE], y_clean = y[keep])
}
res <- removeOutliers(X_clean, y, 2)
X_clean <- res$X_clean
y_clean <- res$y_clean

## 3. Feature Transformation (z-score normalization)
zscoreNormalize <- function(X) {
  mu <- apply(X, 1, mean)
  sigma <- apply(X, 1, sd)
  X_norm <- (X - matrix(mu, nrow = nrow(X), ncol = ncol(X), byrow = FALSE)) /
    matrix(sigma, nrow = nrow(X), ncol = ncol(X), byrow = FALSE)
  list(X_norm = X_norm, mu = mu, sigma = sigma)
}
norm_res <- zscoreNormalize(X_clean)
X_norm <- norm_res$X_norm

## 4. Feature Extraction: PCA
# Transpose data so that observations are rows.
pca_res <- prcomp(t(X_norm), center = FALSE, scale. = FALSE)
# Select the first 2 principal components (transpose back to 2 x N)
X_pca <- t(pca_res$x[, 1:2])

## 5. Feature Derivation: Create polynomial features (degree 2)
createPolynomialFeatures <- function(X, degree = 2) {
  # X: original features (d x N)
  d <- nrow(X); N <- ncol(X)
  X_poly <- X  # start with original features
  if (degree >= 2) {
    # Add square terms
    for (i in 1:d) {
      X_poly <- rbind(X_poly, X[i, ]^2)
    }
    # Add pairwise interaction terms
    if(d > 1) {
      for (i in 1:(d-1)) {
        for (j in (i+1):d) {
          X_poly <- rbind(X_poly, X[i, ] * X[j, ])
        }
      }
    }
  }
  X_poly
}
X_poly <- createPolynomialFeatures(X_norm, 2)

## 6. Visualization
par(mfrow = c(1, 2))
plot(t(X), col = ifelse(y, "blue", "red"), pch = 16,
     xlab = "Feature 1", ylab = "Feature 2", main = "Original Data")
plot(t(X_clean), col = ifelse(y_clean, "blue", "red"), pch = 16,
     xlab = "Feature 1", ylab = "Feature 2", main = "Cleaned Data")

par(mfrow = c(1, 2))
plot(X_pca[1, ], X_pca[2, ], col = ifelse(y_clean, "blue", "red"), pch = 16,
     xlab = "PC1", ylab = "PC2", main = "PCA Extracted Features")
# For visualization, plot the first two polynomial features
plot(X_poly[1, ], X_poly[2, ], col = ifelse(y_clean, "blue", "red"), pch = 16,
     xlab = "Poly Feature 1", ylab = "Poly Feature 2", main = "Polynomial Derived Features")
