# gan_2layer_tutorial.R
# A from-scratch implementation of a simple GAN (using one-layer networks)
# using the built-in "cameraman.tif" image.
#
# This script loads the "cameraman.tif" image, normalizes it, and flattens it into a
# row vector. It then trains a GAN using one-layer networks for both the generator and
# discriminator. (Note: learning rates and number of epochs are different from gan_1layer_tutorial.)
#
# After training, the generator is used to create new images that are displayed alongside
# the real image.

rm(list = ls())
graphics.off()
set.seed(1)
library(tiff)

## Load and Preprocess the Image
I <- readTIFF("data/cameraman.tif", native = FALSE)
I <- im2double(I)  # Normalize image to [0,1]
H <- nrow(I); W <- ncol(I)
imgSize <- c(H, W)
inputDim <- H * W
x_real <- as.vector(I)

## GAN Training Parameters
zDim <- 100         # Dimension of noise vector
epochs <- 1000      # Number of training epochs
alphaD <- 0.0001    # Learning rate for discriminator
alphaG <- 0.0001    # Learning rate for generator
batchSize <- 32     # Minibatch size

## Train the GAN (using one-layer networks as before)
ganTrainCameraman <- function(realData, inputDim, zDim, epochs, alphaD, alphaG, batchSize, imgSize) {
  w_D <- matrix(rnorm(inputDim, sd = 0.01), nrow = 1)
  b_D <- 0
  w_G <- matrix(rnorm(inputDim * zDim, sd = 0.01), nrow = inputDim)
  b_G <- rep(0, inputDim)
  
  sigmoid <- function(x) { 1 / (1 + exp(-x)) }
  
  for (epoch in 1:epochs) {
    realBatch <- matrix(rep(realData, batchSize), nrow = batchSize, byrow = TRUE)
    Z <- matrix(rnorm(zDim * batchSize), nrow = zDim)
    fakeData <- t(w_G %*% Z + matrix(b_G, nrow = inputDim, ncol = batchSize))
    D_real <- sigmoid(w_D %*% t(realBatch) + b_D)
    D_real <- as.vector(D_real)
    D_fake <- sigmoid(w_D %*% t(fakeData) + b_D)
    D_fake <- as.vector(D_fake)
    loss_D <- -mean(log(D_real + 1e-8) + log(1 - D_fake + 1e-8))
    grad_wD <- (colMeans((1 - D_real) * realBatch) - colMeans(D_fake * fakeData)) / 2
    grad_bD <- (mean(1 - D_real) - mean(D_fake)) / 2
    w_D <- w_D - alphaD * matrix(grad_wD, nrow = 1)
    b_D <- b_D - alphaD * grad_bD
    Z <- matrix(rnorm(zDim * batchSize), nrow = zDim)
    fakeData <- t(w_G %*% Z + matrix(b_G, nrow = inputDim, ncol = batchSize))
    D_fake <- sigmoid(w_D %*% t(fakeData) + b_D)
    D_fake <- as.vector(D_fake)
    loss_G <- -mean(log(D_fake + 1e-8))
    grad_output <- (1 - D_fake)
    grad_w_G <- matrix(0, nrow = inputDim, ncol = zDim)
    for (i in 1:batchSize) {
      grad_w_G <- grad_w_G + grad_output[i] * matrix(Z[, i], nrow = inputDim, ncol = zDim, byrow = TRUE)
    }
    grad_w_G <- grad_w_G / batchSize
    grad_b_G <- rep(mean(grad_output), inputDim)
    w_G <- w_G - alphaG * grad_w_G
    b_G <- b_G - alphaG * grad_b_G
    if (epoch %% 100 == 0) {
      cat(sprintf("Epoch %d, Loss_D: %.4f, Loss_G: %.4f\n", epoch, loss_D, loss_G))
    }
  }
  list(w_D = w_D, b_D = b_D, w_G = w_G, b_G = b_G, noiseDim = zDim, imgSize = imgSize)
}

model <- ganTrainCameraman(x_real, inputDim, zDim, epochs, alphaD, alphaG, batchSize, imgSize)

ganPredict <- function(model, numSamples) {
  sigmoid <- function(x) { 1 / (1 + exp(-x)) }
  z <- matrix(rnorm(model$noiseDim * numSamples), nrow = model$noiseDim)
  fakeFlat <- model$w_G %*% z + matrix(model$b_G, nrow = model$w_G %>% nrow, ncol = numSamples, byrow = FALSE)
  fakeFlat <- sigmoid(fakeFlat)
  fakeFlat <- t(fakeFlat)
  H_img <- model$imgSize[1]
  W_img <- model$imgSize[2]
  array(fakeFlat, dim = c(H_img, W_img, numSamples))
}

numGenSamples <- 4
generatedImages <- ganPredict(model, numGenSamples)

## Visualization
par(mfrow = c(1, numGenSamples + 1))
plot(as.cimg(I), main = "Real Image")
for (i in 1:numGenSamples) {
  img_gen <- generatedImages[,, i]
  plot(as.cimg(img_gen), main = paste("Generated", i))
}
