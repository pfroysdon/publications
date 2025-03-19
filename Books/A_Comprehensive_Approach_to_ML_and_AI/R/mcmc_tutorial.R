# MCMC Tutorial: Metropolis–Hastings Sampling from a Standard Normal Distribution in R
#
# This tutorial demonstrates how to implement the Metropolis–Hastings algorithm
# from scratch. We sample from the target distribution:
#
#       p(x) ~ exp(–x^2 / 2)   (a standard normal, up to a constant)
#
# A Gaussian proposal with standard deviation proposalStd is used.

set.seed(1)

# Parameters
numSamples <- 10000   # Total number of samples
burnIn <- 1000        # Number of burn–in samples to discard
proposalStd <- 1.0    # Standard deviation of the proposal distribution

# Target density function (unnormalized standard normal)
targetDensity <- function(x) {
  exp(-0.5 * x^2)
}

# Metropolis–Hastings algorithm
metropolisHastings <- function(targetFunc, numSamples, proposalStd) {
  samples <- numeric(numSamples)
  samples[1] <- 0  # Initialize chain at 0
  for (i in 2:numSamples) {
    current <- samples[i - 1]
    proposal <- current + rnorm(1, mean = 0, sd = proposalStd)
    p_current <- targetFunc(current)
    p_proposal <- targetFunc(proposal)
    acceptance <- min(1, p_proposal / p_current)
    if (runif(1) < acceptance) {
      samples[i] <- proposal
    } else {
      samples[i] <- current
    }
  }
  return(samples)
}

# Run the sampler and remove burn–in samples
samples <- metropolisHastings(targetDensity, numSamples, proposalStd)
samples <- samples[(burnIn + 1):numSamples]

# Visualization: Plot histogram vs. true normal PDF
hist(samples, breaks = 50, probability = TRUE, col = "lightblue",
     main = "MCMC Sampling using Metropolis–Hastings", xlab = "x", ylab = "Probability Density")
x_vals <- seq(min(samples) - 1, max(samples) + 1, length.out = 100)
lines(x_vals, dnorm(x_vals, mean = 0, sd = 1), col = "red", lwd = 2)
legend("topright", legend = c("MCMC Samples", "True Normal PDF"), col = c("lightblue", "red"), lty = c(1, 1), lwd = c(NA, 2))
grid()
