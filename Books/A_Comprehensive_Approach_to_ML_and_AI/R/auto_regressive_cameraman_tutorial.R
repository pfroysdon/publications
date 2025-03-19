# auto_regressive_cameraman_tutorial.R
# Auto-Regressive Deep Learning on the Cameraman Image in R
# This script loads the cameraman image, downsamples it, flattens it into a sequence,
# creates training samples from a sliding window, trains a simple one-hidden-layer neural network,
# and then auto-regressively generates an image pixel by pixel.

auto_regressive_deep_learning_cameraman <- function() {
  library(imager)
  set.seed(1)
  
  # 1. Load and downsample image
  img <- load.image("data/cameraman.tif")
  img <- im2double(img)
  img_ds <- resize(img, size_x = 64, size_y = 64)
  x_full <- as.vector(img_ds)
  N <- length(x_full)
  
  # 2. Create training data: for each index i, use previous window_size pixels to predict x[i+window_size]
  window_size <- 16
  num_samples <- N - window_size
  X_train <- matrix(0, nrow = window_size, ncol = num_samples)
  y_train <- numeric(num_samples)
  for (i in 1:num_samples) {
    X_train[, i] <- x_full[i:(i + window_size - 1)]
    y_train[i] <- x_full[i + window_size]
  }
  
  # 3. Set up neural network parameters: input layer = window_size, hidden layer = 50 neurons, output = 1
  input_dim <- window_size
  hidden_dim <- 50
  set.seed(1)
  W1 <- matrix(rnorm(hidden_dim * input_dim, sd = 0.01), nrow = hidden_dim)
  b1 <- matrix(0, nrow = hidden_dim, ncol = 1)
  W2 <- matrix(rnorm(1 * hidden_dim, sd = 0.01), nrow = 1)
  b2 <- 0
  
  # Activation functions
  sigmoid <- function(z) { 1 / (1 + exp(-z)) }
  relu <- function(z) { pmax(0, z) }
  
  # 4. Train using batch gradient descent
  learning_rate <- 0.01
  num_epochs <- 5000
  m <- num_samples
  losses <- numeric(num_epochs)
  
  for (epoch in 1:num_epochs) {
    Z1 <- W1 %*% X_train + matrix(rep(b1, m), nrow = hidden_dim)
    A1 <- relu(Z1)
    Z2 <- W2 %*% A1 + b2
    A2 <- sigmoid(Z2)
    loss <- 0.5 * mean((A2 - matrix(y_train, nrow = 1))^2)
    losses[epoch] <- loss
    
    dZ2 <- (A2 - matrix(y_train, nrow = 1)) * (sigmoid(Z2) * (1 - sigmoid(Z2)))
    dW2 <- dZ2 %*% t(A1) / m
    db2 <- mean(dZ2)
    dA1 <- t(W2) %*% dZ2
    dZ1 <- dA1 * (ifelse(Z1 > 0, 1, 0))
    dW1 <- dZ1 %*% t(X_train) / m
    db1 <- rowMeans(dZ1)
    
    W1 <- W1 - learning_rate * dW1
    b1 <- b1 - learning_rate * matrix(db1, ncol = 1)
    W2 <- W2 - learning_rate * dW2
    b2 <- b2 - learning_rate * db2
    
    if (epoch %% 500 == 0) {
      cat(sprintf("Epoch %d, Loss: %.6f\n", epoch, loss))
    }
  }
  
  # 5. Auto-regressive generation: seed with first window_size pixels and generate full sequence
  gen_length <- N
  generated_seq <- numeric(gen_length)
  generated_seq[1:window_size] <- x_full[1:window_size]
  for (i in (window_size + 1):gen_length) {
    input_seq <- generated_seq[(i - window_size):(i - 1)]
    z1 <- W1 %*% matrix(input_seq, ncol = 1) + b1
    a1 <- relu(z1)
    z2 <- W2 %*% a1 + b2
    a2 <- sigmoid(z2)
    generated_seq[i] <- a2
  }
  
  gen_img <- matrix(generated_seq, nrow = dim(img_ds)[1], ncol = dim(img_ds)[2])
  
  # 6. Visualize results
  par(mfrow = c(1,2))
  plot(as.raster(img_ds), main = "Original Downsampled Image")
  plot(as.raster(gen_img), main = "Auto-Regressive Generated Image")
  
  par(mfrow = c(1,1))
  plot(1:num_epochs, losses, type = "l", lwd = 2, xlab = "Epoch", ylab = "Loss", main = "Training Loss")
  grid()
}

auto_regressive_deep_learning_cameraman()
