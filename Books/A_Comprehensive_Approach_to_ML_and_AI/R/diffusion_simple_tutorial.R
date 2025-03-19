# diffusion_simple_tutorial.R
# This script demonstrates a simple diffusion process on 1D Gaussian samples.
# It applies a forward diffusion (adding noise) and a reverse diffusion (denoising) process.

rm(list=ls())
graphics.off()
set.seed(42)

X <- matrix(rnorm(100), ncol = 1)  # 100 samples, 1 feature

num_steps <- 50
beta <- 0.02

# Forward diffusion process
forward_diffusion <- function(X, num_steps, beta) {
  X_noisy <- X
  for (t in 1:num_steps) {
    noise <- sqrt(beta) * matrix(rnorm(length(X)), ncol = 1)
    X_noisy <- sqrt(1 - beta) * X_noisy + noise
  }
  X_noisy
}

# Reverse diffusion process using a simple denoising model
reverse_diffusion <- function(X_noisy, num_steps, beta, model) {
  X_denoised <- X_noisy
  for (t in num_steps:1) {
    predicted_noise <- model(X_denoised)
    X_denoised <- (X_denoised - sqrt(beta) * predicted_noise) / sqrt(1 - beta)
  }
  X_denoised
}

X_noisy <- forward_diffusion(X, num_steps, beta)
# Simple denoising model: scale input by 0.9
model <- function(x) { 0.9 * x }
X_denoised <- reverse_diffusion(X_noisy, num_steps, beta, model)

# Plot results
plot(X, col = "blue", pch = 16, main = "Diffusion Process", xlab = "Sample Index", ylab = "Value")
points(X_noisy, col = "red", pch = 16)
points(X_denoised, col = "green", pch = 16)
legend("topright", legend = c("Original", "Noised", "Denoised"),
       col = c("blue", "red", "green"), pch = 16)
