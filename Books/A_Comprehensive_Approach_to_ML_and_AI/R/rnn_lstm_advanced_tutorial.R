# Advanced Climate Forecasting with Dense, RNN, and LSTM Models in R
#
# This tutorial uses the Jena climate dataset to forecast temperature.
# It performs the following steps:
#   1. Loads and parses the data.
#   2. Windows the data to create training, validation, and test sets.
#   3. Normalizes the data using training-set statistics.
#   4. Computes a baseline MAE.
#   5. Trains a densely connected model, an RNN model, and an LSTM model using keras.
#   6. Plots and compares predictions on the validation set.
#
# Note: This example uses the keras package. Install and configure keras beforehand.

library(keras)
library(tidyverse)

# 1. Load and Parse the Data
filename <- "data/jena_climate_2009_2016.csv"
if (!file.exists(filename)) stop("File not found. Please download jena_climate_2009_2016.csv.")
data <- read_csv(filename)
dt <- as.POSIXct(data[[1]], format="%Y-%m-%d %H:%M:%S")
temp <- data[[3]]
cat(sprintf("Loaded %d temperature samples from %s to %s\n", length(temp), as.character(dt[1]), as.character(dt[length(dt)])))

# 2. Prepare the Data (Windowing)
lookback <- 720  # number of past timesteps (~5 days if 10-min intervals)
delay <- 144     # predict 144 timesteps ahead (~1 day)
step <- 6        # sample every 6 timesteps (hourly)
start_index <- 1
end_index <- 200000  # adjust based on data length

create_dataset <- function(data, lookback, delay, step, start_index, end_index) {
  num_samples <- floor((end_index - start_index - lookback - delay) / 1) + 1
  num_timesteps <- floor(lookback / step)
  X <- array(0, dim = c(num_samples, num_timesteps, 1))
  y <- numeric(num_samples)
  for (i in 1:num_samples) {
    idx <- start_index + i - 1
    indices <- seq(idx, idx + lookback - 1, by = step)
    X[i,,1] <- data[indices]
    y[i] <- data[idx + lookback + delay - 1]
  }
  list(X = X, y = y)
}

dataset <- create_dataset(temp, lookback, delay, step, start_index, end_index)
X_train <- dataset$X
y_train <- dataset$y

# For validation and test, use later parts of the data.
val_dataset <- create_dataset(temp, lookback, delay, step, 200001, 300000)
X_val <- val_dataset$X
y_val <- val_dataset$y

# 3. Normalize the Data using training statistics
train_mean <- mean(X_train)
train_std <- sd(X_train)
X_train <- (X_train - train_mean) / train_std
X_val <- (X_val - train_mean) / train_std
y_train <- (y_train - train_mean) / train_std
y_val <- (y_val - train_mean) / train_std

# 4. Baseline: Predict last value in window as forecast.
y_pred_baseline <- X_val[, dim(X_val)[2], 1]
baseline_MAE <- mean(abs(y_pred_baseline - y_val))
cat(sprintf("Baseline MAE (normalized): %.4f\n", baseline_MAE))

# 5. Define and Train Models using Keras

# Dense Model
dense_model <- keras_model_sequential() %>%
  layer_flatten(input_shape = dim(X_train)[2:3]) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 1)
dense_model %>% compile(optimizer = optimizer_adam(lr = 1e-3), loss = "mse")
history_dense <- dense_model %>% fit(X_train, y_train, epochs = 10, batch_size = 32,
                                      validation_data = list(X_val, y_val))
y_pred_dense <- dense_model %>% predict(X_val)

# RNN Model
rnn_model <- keras_model_sequential() %>%
  layer_simple_rnn(units = 32, input_shape = dim(X_train)[2:3]) %>%
  layer_dense(units = 1)
rnn_model %>% compile(optimizer = optimizer_adam(lr = 1e-3), loss = "mse")
history_rnn <- rnn_model %>% fit(X_train, y_train, epochs = 10, batch_size = 32,
                                 validation_data = list(X_val, y_val))
y_pred_rnn <- rnn_model %>% predict(X_val)

# LSTM Model
lstm_model <- keras_model_sequential() %>%
  layer_lstm(units = 32, input_shape = dim(X_train)[2:3]) %>%
  layer_dense(units = 1)
lstm_model %>% compile(optimizer = optimizer_adam(lr = 1e-3), loss = "mse")
history_lstm <- lstm_model %>% fit(X_train, y_train, epochs = 10, batch_size = 32,
                                   validation_data = list(X_val, y_val))
y_pred_lstm <- lstm_model %>% predict(X_val)

# 6. Plot Predictions on Validation Set
t_val <- 1:length(y_val)
plot(t_val, y_val, type = "l", col = "black", lwd = 1.5, xlab = "Sample", ylab = "Normalized Temperature",
     main = "Model Predictions on Validation Set")
lines(t_val, y_pred_baseline, col = "blue")
lines(t_val, y_pred_dense, col = "red")
lines(t_val, y_pred_rnn, col = "green")
lines(t_val, y_pred_lstm, col = "magenta")
legend("topright", legend = c("True", "Baseline", "Dense", "RNN", "LSTM"),
       col = c("black", "blue", "red", "green", "magenta"), lty = 1)
