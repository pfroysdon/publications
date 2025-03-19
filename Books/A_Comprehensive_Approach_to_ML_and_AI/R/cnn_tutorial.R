# cnn_tutorial.R
# This script demonstrates a simple CNN forward pass on the "cameraman.tif" image.
# It performs convolution, ReLU activation, max pooling, flattening, and a fully connected layer.

rm(list=ls())
graphics.off()
library(tiff)
library(imager)

# Load and preprocess image
I <- readTIFF("data/cameraman.tif", native = FALSE)
# (Assuming the image is already normalized to [0,1])
# Display the original image
par(mfrow = c(1, 3))
plot(as.cimg(I), main = "Original Image")

# CNN Parameters
K <- matrix(c(-1, -1, -1, 0, 0, 0, 1, 1, 1), nrow = 3, byrow = TRUE)
poolSize <- 2

# Convolution function (valid convolution)
myConv2 <- function(I, K) {
  H <- nrow(I)
  W <- ncol(I)
  kH <- nrow(K)
  kW <- ncol(K)
  outH <- H - kH + 1
  outW <- W - kW + 1
  S <- matrix(0, nrow = outH, ncol = outW)
  for (i in 1:outH) {
    for (j in 1:outW) {
      patch <- I[i:(i+kH-1), j:(j+kW-1)]
      S[i, j] <- sum(patch * K)
    }
  }
  return(S)
}

# Max pooling function
maxPool <- function(A, poolSize) {
  H <- nrow(A)
  W <- ncol(A)
  Hp <- floor(H / poolSize)
  Wp <- floor(W / poolSize)
  P <- matrix(0, nrow = Hp, ncol = Wp)
  for (i in 1:Hp) {
    for (j in 1:Wp) {
      patch <- A[((i-1)*poolSize+1):(i*poolSize), ((j-1)*poolSize+1):(j*poolSize)]
      P[i, j] <- max(patch)
    }
  }
  return(P)
}

# Forward Pass through CNN
S <- myConv2(I, K)         # Convolution (valid mode)
A <- pmax(S, 0)            # ReLU Activation
P <- maxPool(A, poolSize)   # Max Pooling
f <- as.vector(P)          # Flatten pooled feature map

# Fully connected layer: random weights and bias for demonstration
W_fc <- matrix(rnorm(length(f), sd = 0.01), nrow = 1)
b_fc <- 0
y <- W_fc %*% f + b_fc

# Visualization of intermediate steps
par(mfrow = c(1, 3))
image(A, col = gray.colors(256), main = "ReLU Activation Output", axes = FALSE)
image(P, col = gray.colors(256), main = "Max Pooling Output", axes = FALSE)

# Also display the convolved image separately
par(mfrow = c(1, 2))
image(I, col = gray.colors(256), main = "Original Image", axes = FALSE)
image(S, col = gray.colors(256), main = "Convolved Image (Valid)", axes = FALSE)

cat(sprintf("The fully connected layer output (prediction) is: %.4f\n", y))
