# Low Rank Adaptation (LoRA) Tutorial in R
#
# In this tutorial, we demonstrate Low Rank Adaptation (LoRA) for adapting a 
# pre-trained linear model to a new regression task.
#
# We assume a pre-trained weight vector (row vector) W0 for a linear 
# model y = W0 * x. For a new task with target weight W_target, instead of 
# retraining the full model, we adapt it by learning a low–rank update:
#
#         W = W0 + B*A
#
# where B (size: 1 x r) and A (size: r x d) are trainable and r << d.
#
# The training objective is to minimize the mean squared error between the model 
# prediction and the target output on new data:
#
#         Loss = mean((y - (W0+B*A)*x)^2)
#
# We update B and A via gradient descent while keeping W0 fixed.

set.seed(1)

# Parameters
d <- 10           # input dimension
N <- 200          # number of samples
r <- 2            # rank of adaptation (low–rank update)
learningRate <- 0.01
numEpochs <- 1000

# Generate Data
# X: d x N matrix (each column is a sample)
X <- matrix(rnorm(d * N), nrow = d, ncol = N)
# Define the ideal (target) weight vector W_target (1 x d)
W_target <- seq(1, 2, length.out = d)
# Generate outputs with some noise
Y <- W_target %*% X + 0.1 * rnorm(N)

# Pre-trained Weight (W0)
W0 <- W_target - 0.5  # W0 is 0.5 less than W_target

# LoRA Initialization: learn B and A such that W0 + B*A approximates W_target.
B <- matrix(rnorm(r, sd = 0.01), nrow = 1)       # 1 x r
A <- matrix(rnorm(r * d, sd = 0.01), nrow = r, ncol = d)  # r x d

lossHistory <- numeric(numEpochs)
for (epoch in 1:numEpochs) {
  # Compute adapted weight: W = W0 + B*A (1 x d)
  W <- W0 + B %*% A
  # Predictions: Yhat = W %*% X (1 x N)
  Yhat <- W %*% X
  # Mean squared error loss
  loss <- mean((Y - Yhat)^2)
  lossHistory[epoch] <- loss
  
  # Gradient with respect to W
  error <- Yhat - Y
  gradW <- (error %*% t(X)) / N  # 1 x d
  
  # Chain rule: gradients for B and A
  gradB <- gradW %*% t(A)        # 1 x r
  gradA <- t(B) %*% gradW        # r x d
  
  # Update parameters
  B <- B - learningRate * gradB
  A <- A - learningRate * gradA
  
  if (epoch %% 100 == 0) {
    cat(sprintf("Epoch %d, Loss: %.4f\n", epoch, loss))
  }
}

# Plot Training Loss
plot(1:numEpochs, lossHistory, type = "l", lwd = 2,
     xlab = "Epoch", ylab = "Mean Squared Error Loss",
     main = "LoRA Adaptation Training Loss")
grid()

# Compare Adapted Weight to Target
W_adapted <- W0 + B %*% A
cat("Pre-trained weight W0:\n"); print(W0)
cat("Target weight W_target:\n"); print(W_target)
cat("Adapted weight W0+B*A:\n"); print(W_adapted)

# Predict on a Test Sample (Optional)
x_test <- matrix(rnorm(d), nrow = d, ncol = 1)
y_pred <- W_adapted %*% x_test
y_true <- W_target %*% x_test
cat(sprintf("Test sample prediction: %.4f (target: %.4f)\n", y_pred, y_true))
