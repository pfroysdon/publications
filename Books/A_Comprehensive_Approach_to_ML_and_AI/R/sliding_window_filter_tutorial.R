# Sliding Window Filter Tutorial in R
#
# This script simulates daily temperature data for one year using a sine wave with noise,
# applies a sliding window (moving average) filter, and plots the original and smoothed data.

set.seed(1)

# 1. Generate Synthetic Daily Temperature Data
days <- 1:365
seasonalTemp <- 15 + 10 * sin(2 * pi * (days - 80) / 365)
noise <- rnorm(length(days), sd = 2)
temperature <- seasonalTemp + noise

# 2. Define Sliding Window Filter (Moving Average)
slidingWindowFilter <- function(y, windowSize) {
  if (windowSize %% 2 == 0) windowSize <- windowSize + 1
  halfWindow <- floor(windowSize / 2)
  N <- length(y)
  y_filtered <- numeric(N)
  for (i in 1:N) {
    idx_start <- max(1, i - halfWindow)
    idx_end <- min(N, i + halfWindow)
    y_filtered[i] <- mean(y[idx_start:idx_end])
  }
  y_filtered
}

windowSize <- 7
smoothedTemperature7 <- slidingWindowFilter(temperature, windowSize)
smoothedTemperature30 <- slidingWindowFilter(temperature, 30)

# 3. Plot Original and Smoothed Data
plot(days, temperature, type = "l", col = "blue", lwd = 1.5,
     xlab = "Day", ylab = "Temperature (°C)", main = "Daily Temperature and 7-Day Moving Average")
lines(days, smoothedTemperature7, col = "red", lwd = 2)
legend("topright", legend = c("Original", "7-Day MA"), col = c("blue", "red"), lty = 1)

plot(days, temperature, type = "l", col = "blue", lwd = 1.5,
     xlab = "Day", ylab = "Temperature (°C)", main = "Temperature with 7-Day and 30-Day Moving Averages")
lines(days, smoothedTemperature7, col = "red", lwd = 2)
lines(days, smoothedTemperature30, col = "black", lwd = 2)
legend("topright", legend = c("Original", "7-Day MA", "30-Day MA"), col = c("blue", "red", "black"), lty = 1)
xlim(c(130,230))
