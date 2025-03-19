# autoencoder_cameraman_tutorial.R
# Autoencoder Tutorial in R for the Cameraman Image
# Loads the cameraman image, downsamples it, flattens it into a vector,
# sets up a single-hidden-layer autoencoder, trains it via gradient descent,
# and then displays the original and reconstructed images and the training loss.

rm(list = ls())
graphics.off()
set.seed(1)
library(imager)

# 1. Load and preprocess the image
img <- load.image("data/cameraman.tif")
img <- im2double(img)
# Downsample image (e.g., to 128x128)
img_ds <- resize(img, size_x = 128, size_y = 128)
H <- dim(img_ds)[1]; W <- dim(img_ds)[2]
imgSize <- c(H, W)
x <- as.vector(img_ds)
input_dim <- length(x)

# 2. Set up autoencoder architecture: one hidden layer (encoder + decoder)
hidden_size <- 64
set.seed(1)
W1 <- matrix(rnorm(hidden_size * input_dim, sd = 0.01), nrow = hidden_size)
b1 <- matrix(0, nrow = hidden_size, ncol = 1)
W2 <- matrix(rnorm(input_dim * hidden_size, sd = 0.01), nrow = input_dim)
b2 <- matrix(0, nrow = input_dim, ncol = 1)

# 3. Training parameters
learning_rate <- 0.1
num_epochs <- 100
losses <- numeric(num_epochs)

# Activation functions
sigmoid <- function(z) { 1 / (1 + exp(-z)) }
sigmoid_deriv <- function(z) {
  s <- sigmoid(z)
  s * (1 - s)
}

# 4. Train the autoencoder using gradient descent
for (epoch in 1:num_epochs) {
  z1 <- W1 %*% x + b1
  a1 <- sigmoid(z1)
  z2 <- W2 %*% a1 + b2
  a2 <- sigmoid(z2)
  
  loss <- 0.5 * sum((a2 - x)^2)
  losses[epoch] <- loss
  
  delta2 <- (a2 - x) * sigmoid_deriv(z2)
  dW2 <- delta2 %*% t(a1)
  db2 <- delta2
  dA1 <- t(W2) %*% delta2
  delta1 <- dA1 * sigmoid_deriv(z1)
  dW1 <- delta1 %*% t(x)
  db1 <- delta1
  
  W1 <- W1 - learning_rate * dW1
  b1 <- b1 - learning_rate * db1
  W2 <- W2 - learning_rate * dW2
  b2 <- b2 - learning_rate * db2
  
  if (epoch %% 10 == 0) {
    cat(sprintf("Epoch %d, Loss: %.4f\n", epoch, loss))
  }
}

# 5. Reconstruct the image
I_recon <- matrix(a2, nrow = H, ncol = W)
I_recon <- as.cimg(I_recon)

# 6. Visualization
par(mfrow = c(1,2))
plot(as.raster(img_ds), main = "Original Downsampled Image")
plot(as.raster(I_recon), main = "Reconstructed Image by Autoencoder")

par(mfrow = c(1,1))
plot(1:num_epochs, losses, type = "l", lwd = 2, xlab = "Epoch", ylab = "Loss", main = "Training Loss")
grid()
