# LSTM Time–Series Prediction Tutorial in R (from scratch)
#
# This tutorial demonstrates how to implement a simple LSTM network for
# time–series prediction without using deep learning libraries.
# We generate a synthetic sine–wave signal with noise, train the LSTM to
# predict the signal, and then plot the true vs. predicted values.

set.seed(1)

# Generate Synthetic Data
T_steps <- 100                           # Number of time steps
t_axis <- seq(0, 2 * pi, length.out = T_steps)  # Time axis
X <- matrix(sin(t_axis), nrow = 1)         # 1 x T input sequence
Y <- sin(t_axis + 0.5) + 0.1 * rnorm(T_steps)  # Target sequence

# LSTM Hyperparameters
hiddenSize <- 10
alpha <- 0.01      # Learning rate
epochs <- 5000     # Number of training epochs

# Define activation functions
sigmoid <- function(x) {
  1 / (1 + exp(-x))
}

# LSTM training function
lstmTrain <- function(X, Y, hiddenSize, alpha, epochs) {
  d <- nrow(X)  # input dimension (here, 1)
  T_steps <- ncol(X)
  outputSize <- 1  # Regression output
  
  # Initialize LSTM parameters with small random values
  Wxi <- matrix(rnorm(hiddenSize * d, sd = 0.01), nrow = hiddenSize)
  Whi <- matrix(rnorm(hiddenSize * hiddenSize, sd = 0.01), nrow = hiddenSize)
  bi  <- matrix(0, nrow = hiddenSize, ncol = 1)
  
  Wxf <- matrix(rnorm(hiddenSize * d, sd = 0.01), nrow = hiddenSize)
  Whf <- matrix(rnorm(hiddenSize * hiddenSize, sd = 0.01), nrow = hiddenSize)
  bf  <- matrix(0, nrow = hiddenSize, ncol = 1)
  
  Wxo <- matrix(rnorm(hiddenSize * d, sd = 0.01), nrow = hiddenSize)
  Who <- matrix(rnorm(hiddenSize * hiddenSize, sd = 0.01), nrow = hiddenSize)
  bo  <- matrix(0, nrow = hiddenSize, ncol = 1)
  
  Wxc <- matrix(rnorm(hiddenSize * d, sd = 0.01), nrow = hiddenSize)
  Whc <- matrix(rnorm(hiddenSize * hiddenSize, sd = 0.01), nrow = hiddenSize)
  bc  <- matrix(0, nrow = hiddenSize, ncol = 1)
  
  # Output layer parameters
  Why <- matrix(rnorm(outputSize * hiddenSize, sd = 0.01), nrow = outputSize)
  by  <- matrix(0, nrow = outputSize, ncol = 1)
  
  for (epoch in 1:epochs) {
    # Forward Pass: Initialize states
    h <- matrix(0, nrow = hiddenSize, ncol = T_steps)
    c <- matrix(0, nrow = hiddenSize, ncol = T_steps)
    h_prev <- matrix(0, nrow = hiddenSize, ncol = 1)
    c_prev <- matrix(0, nrow = hiddenSize, ncol = 1)
    y_pred <- numeric(T_steps)
    
    # Store activations for BPTT
    i_store <- matrix(0, nrow = hiddenSize, ncol = T_steps)
    f_store <- matrix(0, nrow = hiddenSize, ncol = T_steps)
    o_store <- matrix(0, nrow = hiddenSize, ncol = T_steps)
    g_store <- matrix(0, nrow = hiddenSize, ncol = T_steps)
    
    for (t in 1:T_steps) {
      x_t <- matrix(X[, t], nrow = d)  # current input (d x 1)
      i_t <- sigmoid(Wxi %*% x_t + Whi %*% h_prev + bi)
      f_t <- sigmoid(Wxf %*% x_t + Whf %*% h_prev + bf)
      o_t <- sigmoid(Wxo %*% x_t + Who %*% h_prev + bo)
      g_t <- tanh(Wxc %*% x_t + Whc %*% h_prev + bc)
      
      c_t <- f_t * c_prev + i_t * g_t
      h_t <- o_t * tanh(c_t)
      
      i_store[, t] <- i_t
      f_store[, t] <- f_t
      o_store[, t] <- o_t
      g_store[, t] <- g_t
      c[, t] <- c_t
      h[, t] <- h_t
      
      y_pred[t] <- (Why %*% h_t + by)[1, 1]
      
      h_prev <- h_t
      c_prev <- c_t
    }
    
    # Compute loss (Mean Squared Error)
    loss <- 0.5 * sum((Y - y_pred)^2)
    
    # Initialize gradients for all parameters
    dWxi <- matrix(0, nrow = hiddenSize, ncol = d)
    dWhi <- matrix(0, nrow = hiddenSize, ncol = hiddenSize)
    dbi  <- matrix(0, nrow = hiddenSize, ncol = 1)
    dWxf <- matrix(0, nrow = hiddenSize, ncol = d)
    dWhf <- matrix(0, nrow = hiddenSize, ncol = hiddenSize)
    dbf  <- matrix(0, nrow = hiddenSize, ncol = 1)
    dWxo <- matrix(0, nrow = hiddenSize, ncol = d)
    dWho <- matrix(0, nrow = hiddenSize, ncol = hiddenSize)
    dbo  <- matrix(0, nrow = hiddenSize, ncol = 1)
    dWxc <- matrix(0, nrow = hiddenSize, ncol = d)
    dWhc <- matrix(0, nrow = hiddenSize, ncol = hiddenSize)
    dbc  <- matrix(0, nrow = hiddenSize, ncol = 1)
    dWhy <- matrix(0, nrow = outputSize, ncol = hiddenSize)
    dby  <- matrix(0, nrow = outputSize, ncol = 1)
    
    dh_next <- matrix(0, nrow = hiddenSize, ncol = 1)
    dc_next <- matrix(0, nrow = hiddenSize, ncol = 1)
    
    # Backpropagation Through Time (BPTT)
    for (t in T_steps:1) {
      dy <- y_pred[t] - Y[t]
      dWhy <- dWhy + dy * t(matrix(h[, t], nrow = hiddenSize))
      dby <- dby + dy
      
      dh <- t(Why) %*% dy + dh_next  # (hiddenSize x 1)
      
      do <- dh * tanh(c[, t])
      do <- do * o_store[, t] * (1 - o_store[, t])
      
      dct <- dh * o_store[, t] * (1 - tanh(c[, t])^2) + dc_next
      
      di <- dct * g_store[, t]
      c_prev_val <- if (t == 1) matrix(0, nrow = hiddenSize, ncol = 1) else matrix(c[, t - 1], nrow = hiddenSize)
      df <- dct * c_prev_val
      dg <- dct * i_store[, t]
      dc_prev <- dct * f_store[, t]
      
      di <- di * i_store[, t] * (1 - i_store[, t])
      df <- df * f_store[, t] * (1 - f_store[, t])
      dg <- dg * (1 - g_store[, t]^2)
      
      x_t <- matrix(X[, t], nrow = d)
      h_prev_t <- if (t == 1) matrix(0, nrow = hiddenSize, ncol = 1) else matrix(h[, t - 1], nrow = hiddenSize)
      
      dWxi <- dWxi + di %*% t(x_t)
      dWhi <- dWhi + di %*% t(h_prev_t)
      dbi <- dbi + di
      
      dWxf <- dWxf + df %*% t(x_t)
      dWhf <- dWhf + df %*% t(h_prev_t)
      dbf <- dbf + df
      
      dWxo <- dWxo + do %*% t(x_t)
      dWho <- dWho + do %*% t(h_prev_t)
      dbo <- dbo + do
      
      dWxc <- dWxc + dg %*% t(x_t)
      dWhc <- dWhc + dg %*% t(h_prev_t)
      dbc <- dbc + dg
      
      dh_next <- t(Whi) %*% di + t(Whf) %*% df + t(Who) %*% do + t(Whc) %*% dg
      dc_next <- dc_prev
    }
    
    # Average gradients over time steps
    dWxi <- dWxi / T_steps; dWhi <- dWhi / T_steps; dbi <- dbi / T_steps
    dWxf <- dWxf / T_steps; dWhf <- dWhf / T_steps; dbf <- dbf / T_steps
    dWxo <- dWxo / T_steps; dWho <- dWho / T_steps; dbo <- dbo / T_steps
    dWxc <- dWxc / T_steps; dWhc <- dWhc / T_steps; dbc <- dbc / T_steps
    dWhy <- dWhy / T_steps; dby <- dby / T_steps
    
    # Parameter updates
    Wxi <- Wxi - alpha * dWxi; Whi <- Whi - alpha * dWhi; bi <- bi - alpha * dbi
    Wxf <- Wxf - alpha * dWxf; Whf <- Whf - alpha * dWhf; bf <- bf - alpha * dbf
    Wxo <- Wxo - alpha * dWxo; Who <- Who - alpha * dWho; bo <- bo - alpha * dbo
    Wxc <- Wxc - alpha * dWxc; Whc <- Whc - alpha * dWhc; bc <- bc - alpha * dbc
    Why <- Why - alpha * dWhy; by <- by - alpha * dby
    
    if (epoch %% 100 == 0) {
      cat(sprintf("Epoch %d, Loss: %.4f\n", epoch, loss))
    }
  }
  
  model <- list(Wxi = Wxi, Whi = Whi, bi = bi,
                Wxf = Wxf, Whf = Whf, bf = bf,
                Wxo = Wxo, Who = Who, bo = bo,
                Wxc = Wxc, Whc = Whc, bc = bc,
                Why = Why, by = by,
                hiddenSize = hiddenSize)
  return(model)
}

# LSTM prediction function
lstmPredict <- function(model, X) {
  d <- nrow(X)
  T_steps <- ncol(X)
  h_prev <- matrix(0, nrow = model$hiddenSize, ncol = 1)
  c_prev <- matrix(0, nrow = model$hiddenSize, ncol = 1)
  y_out <- numeric(T_steps)
  for (t in 1:T_steps) {
    x_t <- matrix(X[, t], nrow = d)
    i_t <- sigmoid(model$Wxi %*% x_t + model$Whi %*% h_prev + model$bi)
    f_t <- sigmoid(model$Wxf %*% x_t + model$Whf %*% h_prev + model$bf)
    o_t <- sigmoid(model$Wxo %*% x_t + model$Who %*% h_prev + model$bo)
    g_t <- tanh(model$Wxc %*% x_t + model$Whc %*% h_prev + model$bc)
    c_t <- f_t * c_prev + i_t * g_t
    h_t <- o_t * tanh(c_t)
    y_out[t] <- (model$Why %*% h_t + model$by)[1, 1]
    h_prev <- h_t
    c_prev <- c_t
  }
  return(y_out)
}

# Train the LSTM
model <- lstmTrain(X, Y, hiddenSize, alpha, epochs)

# Predict using the trained LSTM
Y_pred <- lstmPredict(model, X)

# Plot True vs. Predicted Time Series
plot(t_axis, Y, type = "l", col = "blue", lwd = 2,
     xlab = "Time", ylab = "Output", main = "LSTM Time–Series Prediction")
lines(t_axis, Y_pred, col = "red", lwd = 2, lty = 2)
legend("topright", legend = c("True Values", "Predicted Values"),
       col = c("blue", "red"), lty = c(1, 2))
grid()
