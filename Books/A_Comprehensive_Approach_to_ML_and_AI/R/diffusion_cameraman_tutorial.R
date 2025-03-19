# diffusion_cameraman_tutorial.R
# This script implements a simple diffusion model on the "cameraman.tif" image.
# It performs a forward diffusion process (adding noise) and a reverse diffusion (reconstruction).

rm(list=ls())
graphics.off()
library(tiff)
library(imager)

# Load and preprocess the image
I <- readTIFF("data/cameraman.tif", native = FALSE)
I <- im2double(as.cimg(I))
H <- dim(I)[1]
W <- dim(I)[2]
x0 <- as.vector(I)

# Diffusion model parameters
T_steps <- 50  # Number of diffusion steps
beta <- seq(0.0001, 0.02, length.out = T_steps)
alpha <- 1 - beta

x_forward <- matrix(0, nrow = length(x0), ncol = T_steps + 1)
x_forward[, 1] <- x0
eps_store <- matrix(0, nrow = length(x0), ncol = T_steps)

# Forward Diffusion Process: add noise iteratively
for (t in 1:T_steps) {
  eps_t <- rnorm(length(x0))
  eps_store[, t] <- eps_t
  x_forward[, t + 1] <- sqrt(alpha[t]) * x_forward[, t] + sqrt(beta[t]) * eps_t
}

# Reverse Diffusion Process: reconstruct using stored noise
x_reverse <- matrix(0, nrow = length(x0), ncol = T_steps + 1)
x_reverse[, T_steps + 1] <- x_forward[, T_steps + 1]
for (t in T_steps:1) {
  x_reverse[, t] <- (x_reverse[, t + 1] - sqrt(beta[t]) * eps_store[, t]) / sqrt(alpha[t])
}

I_recon <- matrix(x_reverse[, 1], nrow = H, ncol = W)
I_noisy <- matrix(x_forward[, T_steps + 1], nrow = H, ncol = W)

par(mfrow = c(1, 3))
image(1:W, 1:H, I, col = gray.colors(256), main = "Original Image", axes = FALSE)
image(1:W, 1:H, I_noisy, col = gray.colors(256), main = "Noisy Image (Forward)", axes = FALSE)
image(1:W, 1:H, I_recon, col = gray.colors(256), main = "Reconstructed Image (Reverse)", axes = FALSE)

# Plot absolute difference between original and reconstructed images
diff_img <- abs(I - I_recon)
image(1:W, 1:H, diff_img, col = gray.colors(256), main = "Absolute Difference", axes = FALSE)
