# Supervised Fine-Tuning (SFT) Tutorial in R
#
# This tutorial simulates supervised fine-tuning. First, a "pre-trained" network is trained
# on a larger synthetic dataset for 2-class classification. Then, a smaller downstream dataset is used
# to fine-tune the network starting from the pre-trained weights. The network architecture is:
#   Input (2 features) -> Hidden layer (10 neurons, ReLU) -> Output (2 neurons, softmax)

set.seed(1)

# 1. Generate Pre-training Dataset
N_pre <- 500
# Class 1: centered at (1,1)
X1_pre <- matrix(rnorm(2 * (N_pre/2), sd = 0.5), nrow = 2) + matrix(rep(c(1,1), N_pre/2), nrow = 2)
# Class 2: centered at (3,3)
X2_pre <- matrix(rnorm(2 * (N_pre/2), sd = 0.5), nrow = 2) + matrix(rep(c(3,3), N_pre/2), nrow = 2)
X_pre <- cbind(X1_pre, X2_pre)
labels_pre <- c(rep(1, N_pre/2), rep(2, N_pre/2))
# One-hot encoding for pre-training labels
Y_pre <- matrix(0, nrow = 2, ncol = N_pre)
for (i in 1:N_pre) {
  Y_pre[labels_pre[i], i] <- 1
}

# 2. Pre-train the Model (2 -> 10 -> 2)
inputDim <- 2; hiddenDim <- 10; outputDim <- 2
W1 <- matrix(rnorm(hiddenDim * inputDim, sd = 0.01), nrow = hiddenDim)
b1 <- matrix(0, nrow = hiddenDim, ncol = 1)
W2 <- matrix(rnorm(outputDim * hiddenDim, sd = 0.01), nrow = outputDim)
b2 <- matrix(0, nrow = outputDim, ncol = 1)
lr_pre <- 0.01
numIter_pre <- 3000
relu <- function(x) pmax(x, 0)
reluDerivative <- function(x) as.numeric(x > 0)
softmax <- function(x) {
  x <- x - max(x)
  exp(x) / sum(exp(x))
}

for (iter in 1:numIter_pre) {
  Z1 <- W1 %*% X_pre + matrix(rep(b1, N_pre), nrow = hiddenDim)
  H <- relu(Z1)
  logits <- W2 %*% H + matrix(rep(b2, N_pre), nrow = outputDim)
  probs <- apply(logits, 2, softmax)
  loss_pre <- -mean(colSums(Y_pre * log(probs + 1e-8)))
  
  d_logits <- probs - Y_pre
  grad_W2 <- (d_logits %*% t(H)) / N_pre
  grad_b2 <- matrix(rowMeans(d_logits), ncol = 1)
  d_H <- t(W2) %*% d_logits
  d_Z1 <- d_H * matrix(reluDerivative(Z1), nrow = hiddenDim)
  grad_W1 <- (d_Z1 %*% t(X_pre)) / N_pre
  grad_b1 <- matrix(rowMeans(d_Z1), ncol = 1)
  
  W2 <- W2 - lr_pre * grad_W2
  b2 <- b2 - lr_pre * grad_b2
  W1 <- W1 - lr_pre * grad_W1
  b1 <- b1 - lr_pre * grad_b1
  
  if (iter %% 500 == 0) {
    cat(sprintf("Pre-training Iteration %d, Loss: %.4f\n", iter, loss_pre))
  }
}

pretrained <- list(W1 = W1, b1 = b1, W2 = W2, b2 = b2)

# 3. Generate Fine-Tuning Dataset (Downstream Task)
N_ft <- 100
X1_ft <- matrix(rnorm(2 * (N_ft/2), sd = 0.5), nrow = 2) + matrix(rep(c(1.5,1.5), N_ft/2), nrow = 2)
X2_ft <- matrix(rnorm(2 * (N_ft/2), sd = 0.5), nrow = 2) + matrix(rep(c(2.5,2.5), N_ft/2), nrow = 2)
X_ft <- cbind(X1_ft, X2_ft)
labels_ft <- c(rep(1, N_ft/2), rep(2, N_ft/2))
Y_ft <- matrix(0, nrow = 2, ncol = N_ft)
for (i in 1:N_ft) {
  Y_ft[labels_ft[i], i] <- 1
}

# 4. Fine-Tuning using Pre-trained Weights
lr_ft <- 0.001
numIter_ft <- 2000
for (iter in 1:numIter_ft) {
  Z1 <- pretrained$W1 %*% X_ft + matrix(rep(pretrained$b1, N_ft), nrow = hiddenDim)
  H <- relu(Z1)
  logits <- pretrained$W2 %*% H + matrix(rep(pretrained$b2, N_ft), nrow = outputDim)
  probs <- apply(logits, 2, softmax)
  loss_ft <- -mean(colSums(Y_ft * log(probs + 1e-8)))
  
  d_logits <- probs - Y_ft
  grad_W2 <- (d_logits %*% t(H)) / N_ft
  grad_b2 <- matrix(rowMeans(d_logits), ncol = 1)
  d_H <- t(pretrained$W2) %*% d_logits
  d_Z1 <- d_H * matrix(reluDerivative(Z1), nrow = hiddenDim)
  grad_W1 <- (d_Z1 %*% t(X_ft)) / N_ft
  grad_b1 <- matrix(rowMeans(d_Z1), ncol = 1)
  
  pretrained$W2 <- pretrained$W2 - lr_ft * grad_W2
  pretrained$b2 <- pretrained$b2 - lr_ft * grad_b2
  pretrained$W1 <- pretrained$W1 - lr_ft * grad_W1
  pretrained$b1 <- pretrained$b1 - lr_ft * grad_b1
  
  if (iter %% 500 == 0) {
    cat(sprintf("Fine-tuning Iteration %d, Loss: %.4f\n", iter, loss_ft))
  }
}

# 5. Visualize Decision Boundaries Before and After Fine-Tuning
x_min <- min(X_ft[1,]) - 1
x_max <- max(X_ft[1,]) + 1
y_min <- min(X_ft[2,]) - 1
y_max <- max(X_ft[2,]) + 1
xGrid <- seq(x_min, x_max, length.out = 100)
yGrid <- seq(y_min, y_max, length.out = 100)
gridPoints <- rbind(as.vector(outer(xGrid, rep(1,100))),
                    as.vector(outer(rep(1,100), yGrid)))

predictFun <- function(W1, b1, W2, b2, X) {
  H <- apply(X, 2, function(x) relu(W1 %*% matrix(x, ncol = 1) + b1))
  logits <- W2 %*% H + matrix(rep(b2, ncol(X)), nrow = nrow(W2))
  apply(logits, 2, function(z) which.max(softmax(z)))
}

# Pre-trained (before fine-tuning) predictions would be saved separately.
# Here we assume the initial weights before fine-tuning were stored.
# For demonstration, we show the fine-tuned decision boundary.
pred_ft <- predictFun(pretrained$W1, pretrained$b1, pretrained$W2, pretrained$b2, gridPoints)
pred_ft_grid <- matrix(pred_ft, nrow = 100, ncol = 100)

# Plot the fine-tuned decision boundary
image(xGrid, yGrid, pred_ft_grid, col = terrain.colors(2), main = "Fine-tuned Model Decision Boundary",
      xlab = "Feature 1", ylab = "Feature 2")
