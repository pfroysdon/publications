# vae_cameraman_tutorial.R
# Variational Autoencoder (VAE) Tutorial in R using the cameraman image.
# This script loads the image, flattens it, trains a simple VAE, and reconstructs the image.
# Note: This is a simplified example with a very rudimentary backpropagation scheme.

set.seed(1)
library(imager)

# 1. Load and Preprocess the Image
I <- load.image("data/cameraman.tif")
I <- as.cimg(I)  # Ensure it's a cimg object
I <- imnormalize(I)  # Normalize to [0,1]
imgSize <- dim(I)[1:2]  # height and width
x <- as.vector(I)  # Flatten image into a vector

# VAE Hyperparameters
inputDim <- length(x)
latentDim <- 20
hiddenDim <- 100
alpha <- 0.001
epochs <- 2000

# Sigmoid function
sigmoid <- function(x) 1 / (1 + exp(-x))

# VAE Training Function (Simplified)
vaeTrain <- function(x, inputDim, latentDim, hiddenDim, alpha, epochs) {
  # Initialize Encoder parameters
  W_enc <- matrix(rnorm(hiddenDim * inputDim, sd = 0.01), nrow = hiddenDim)
  b_enc <- matrix(0, nrow = hiddenDim, ncol = 1)
  W_mu <- matrix(rnorm(latentDim * hiddenDim, sd = 0.01), nrow = latentDim)
  b_mu <- matrix(0, nrow = latentDim, ncol = 1)
  W_logvar <- matrix(rnorm(latentDim * hiddenDim, sd = 0.01), nrow = latentDim)
  b_logvar <- matrix(0, nrow = latentDim, ncol = 1)
  
  # Initialize Decoder parameters
  W_dec <- matrix(rnorm(hiddenDim * latentDim, sd = 0.01), nrow = hiddenDim)
  b_dec <- matrix(0, nrow = hiddenDim, ncol = 1)
  W_out <- matrix(rnorm(inputDim * hiddenDim, sd = 0.01), nrow = inputDim)
  b_out <- matrix(0, nrow = inputDim, ncol = 1)
  
  for (epoch in 1:epochs) {
    # Encoder Forward Pass
    h_enc <- tanh(W_enc %*% x + b_enc)
    mu <- W_mu %*% h_enc + b_mu
    logvar <- W_logvar %*% h_enc + b_logvar
    sigma <- exp(0.5 * logvar)
    
    # Reparameterization Trick
    epsilon <- matrix(rnorm(latentDim), ncol = 1)
    z <- mu + sigma * epsilon
    
    # Decoder Forward Pass
    h_dec <- tanh(W_dec %*% z + b_dec)
    x_hat <- sigmoid(W_out %*% h_dec + b_out)
    
    # Loss Computation
    recon_loss <- -sum(x * log(x_hat + 1e-8) + (1 - x) * log(1 - x_hat + 1e-8))
    kl_loss <- -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    loss <- recon_loss + kl_loss
    
    # Simplified Backpropagation for output layer only (for demonstration)
    dL_dxhat <- x_hat - x
    grad_W_out <- dL_dxhat %*% t(h_dec)
    grad_b_out <- dL_dxhat
    
    # Update decoder output parameters
    W_out <- W_out - alpha * grad_W_out
    b_out <- b_out - alpha * grad_b_out
    
    if (epoch %% 200 == 0) {
      cat(sprintf("Epoch %d, Loss: %.4f (Recon: %.4f, KL: %.4f)\n", epoch, loss, recon_loss, kl_loss))
    }
  }
  
  list(W_enc = W_enc, b_enc = b_enc, W_mu = W_mu, b_mu = b_mu,
       W_logvar = W_logvar, b_logvar = b_logvar,
       W_dec = W_dec, b_dec = b_dec, W_out = W_out, b_out = b_out)
}

# VAE Prediction Function
vaePredict <- function(model, x) {
  h_enc <- tanh(model$W_enc %*% x + model$b_enc)
  mu <- model$W_mu %*% h_enc + model$b_mu
  # Use mu as latent representation (no sampling)
  z <- mu
  h_dec <- tanh(model$W_dec %*% z + model$b_dec)
  x_hat <- sigmoid(model$W_out %*% h_dec + model$b_out)
  x_hat
}

# Train the VAE
model <- vaeTrain(x, inputDim, latentDim, hiddenDim, alpha, epochs)

# Reconstruct the image using the trained VAE
x_recon <- vaePredict(model, x)
I_recon <- as.cimg(matrix(x_recon, nrow = imgSize[1], ncol = imgSize[2]))

# Visualization
par(mfrow = c(1,3))
plot(I, main = "Original Image")
plot(I_recon, main = "Reconstructed Image")
image(abs(I - I_recon), col = gray.colors(256), main = "Absolute Difference")
