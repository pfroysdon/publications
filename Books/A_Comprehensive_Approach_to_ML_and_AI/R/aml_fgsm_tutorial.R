# aml_fgsm_tutorial.R
# Fast Gradient Sign Method (FGSM) Adversarial Attack Tutorial in R
# Loads the "cameraman.tif" image, defines a logistic regression classifier,
# computes the gradient of the loss with respect to the input,
# and generates an adversarial example.

set.seed(1)
library(imager)

# Load and preprocess the image
I <- load.image("data/cameraman.tif")
I <- im2double(I)
H <- dim(I)[1]; W <- dim(I)[2]
inputDim <- H * W
x <- as.vector(I)
x <- t(x)  # 1 x inputDim

# Define simple logistic regression classifier: f(x) = sigmoid(w*x' + b)
w <- rep(0.001, inputDim)
b <- -0.5
sigmoid <- function(z) { 1 / (1 + exp(-z)) }
classifier <- function(x) { sigmoid(sum(w * x) + b) }

y_orig <- classifier(as.numeric(x))
cat(sprintf("Classifier output on original image: %.4f\n", y_orig))

# Compute loss L = -log(f(x))
loss <- -log(y_orig + 1e-8)

# Compute gradient: grad = (f(x)-1)*w
grad <- (y_orig - 1) * w

# FGSM update: x_adv = x + epsilon * sign(grad)
epsilon <- 0.1
x_adv <- as.numeric(x) + epsilon * sign(grad)

y_adv <- classifier(x_adv)
cat(sprintf("Classifier output on adversarial image: %.4f\n", y_adv))

I_adv <- as.cimg(matrix(x_adv, nrow = H, ncol = W))

par(mfrow = c(1,2))
plot(I, main = "Original Image")
plot(I_adv, main = "Adversarial Image (FGSM)")
