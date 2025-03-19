# aml_pgd_tutorial.R
# Projected Gradient Descent (PGD) Adversarial Attack Tutorial in R
# Uses a logistic regression classifier on the "cameraman.tif" image and applies PGD to create an adversarial example.

set.seed(1)
library(imager)

# Load and preprocess image
I <- load.image("data/cameraman.tif")
I <- im2double(I)
H <- dim(I)[1]; W <- dim(I)[2]
inputDim <- H * W
x <- as.vector(I)
x <- t(x)

# Define classifier: f(x) = sigmoid(w*x' + b)
w <- rep(0.001, inputDim)
b <- -0.5
sigmoid <- function(z) { 1 / (1 + exp(-z)) }
classifier <- function(x) { sigmoid(sum(w * x) + b) }

y_orig <- classifier(as.numeric(x))
cat(sprintf("Classifier output on original image: %.4f\n", y_orig))

# PGD parameters
epsilon <- 0.1
alpha_step <- 0.01
numSteps <- 10

# Numerical gradient (central differences)
numericalGradient <- function(f, x) {
  h <- 1e-5
  grad <- numeric(length(x))
  for (i in 1:length(x)) {
    e <- rep(0, length(x))
    e[i] <- h
    grad[i] <- (f(x + e) - f(x - e)) / (2 * h)
  }
  grad
}

# PGD attack function
pgdAttack <- function(x, classifier, epsilon, alpha, numSteps) {
  x_adv <- as.numeric(x)
  for (t in 1:numSteps) {
    loss_fn <- function(x_val) { -log(classifier(x_val) + 1e-8) }
    grad <- numericalGradient(loss_fn, x_adv)
    x_adv <- x_adv + alpha * sign(grad)
    x_adv <- pmax(x_adv, as.numeric(x) - epsilon)
    x_adv <- pmin(x_adv, as.numeric(x) + epsilon)
    x_adv <- pmax(x_adv, 0)
    x_adv <- pmin(x_adv, 1)
  }
  x_adv
}

x_adv <- pgdAttack(x, classifier, epsilon, alpha_step, numSteps)
y_adv <- classifier(x_adv)
cat(sprintf("Classifier output on adversarial image: %.4f\n", y_adv))

I_adv <- as.cimg(matrix(x_adv, nrow = H, ncol = W))
par(mfrow = c(1,2))
plot(I, main = "Original Image")
plot(I_adv, main = "Adversarial Image (PGD)")
