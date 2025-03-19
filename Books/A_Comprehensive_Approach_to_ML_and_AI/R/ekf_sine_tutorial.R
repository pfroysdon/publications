# ekf_sine_tutorial.R
# Extended Kalman Filter (EKF) for 2D Tracking with a Sinusoidal Trajectory
#
# In this tutorial, we simulate a target moving in 2D with a sinusoidal trajectory.
# The horizontal position increases linearly and the vertical position follows a sine wave:
#
#     x(t) = v_x * t,   y(t) = A * sin(omega * t)
#     vx = v_x,         vy = A * omega * cos(omega * t)
#
# Noisy measurements (range and bearing from the origin) are generated.
# The EKF uses a constant velocity process model.
#
# Changes to improve filter stability:
#   - Process noise covariance Q is increased to better account for model mismatch.
#   - Initial covariance P is set to a larger value.

rm(list = ls())
graphics.off()
set.seed(1)
library(MASS)   # for mvrnorm

## Simulation Parameters
dt <- 0.1           # Time step (s)
T_total <- 20       # Total simulation time (s)
t <- seq(0, T_total, by = dt)   # Time vector
N <- length(t)      # Number of time steps

## True State Simulation (Sinusoidal Trajectory)
# State vector: [x; y; vx; vy]
v_x <- 1.0          # Constant horizontal velocity (m/s)
A <- 2.0            # Amplitude for vertical sinusoid (m)
omega <- 0.5        # Angular frequency (rad/s)
x_true <- matrix(0, nrow = 4, ncol = N)
for (k in 1:N) {
  x_true[1, k] <- v_x * t[k]                   # x position
  x_true[2, k] <- A * sin(omega * t[k])          # y position (sinusoidal)
  x_true[3, k] <- v_x                          # x velocity
  x_true[4, k] <- A * omega * cos(omega * t[k])  # y velocity
}

## Measurement Simulation
# Measurement model: z = [range; bearing]
R_meas <- matrix(c(0.1^2, 0, 0, (pi/180*5)^2), nrow = 2)  # Measurement noise covariance
z <- matrix(0, nrow = 2, ncol = N)
for (k in 1:N) {
  pos <- x_true[1:2, k]
  range <- sqrt(sum(pos^2))
  bearing <- atan2(pos[2], pos[1])
  noise <- as.numeric(mvrnorm(1, mu = c(0, 0), Sigma = R_meas))
  z[, k] <- c(range, bearing) + noise
}

## EKF Initialization
# Process model: constant velocity model
F <- matrix(c(1, 0, dt, 0,
              0, 1, 0, dt,
              0, 0, 1, 0,
              0, 0, 0, 1), nrow = 4, byrow = TRUE)
P <- 10 * diag(4)                # Initial state covariance
Q <- diag(c(0.005, 0.005, 0.05, 0.05))  # Process noise covariance

# Initialize state estimate (slightly off true state)
x_est <- matrix(0, nrow = 4, ncol = N)
x_est[, 1] <- c(0.2, -0.2, 0.8, 0.6)

## Extended Kalman Filter Implementation
for (k in 1:(N - 1)) {
  ## Prediction Step
  x_pred <- F %*% x_est[, k]
  P_pred <- F %*% P %*% t(F) + Q
  
  ## Measurement Prediction
  x_pos <- x_pred[1]
  y_pos <- x_pred[2]
  range_pred <- sqrt(x_pos^2 + y_pos^2)
  bearing_pred <- atan2(y_pos, x_pos)
  z_pred <- c(range_pred, bearing_pred)
  
  ## Compute Jacobian of h(x) at x_pred
  if (range_pred < 1e-4) {
    H_jacobian <- matrix(0, nrow = 2, ncol = 4)
  } else {
    d_range_dx <- x_pos / range_pred
    d_range_dy <- y_pos / range_pred
    d_bearing_dx <- -y_pos / (range_pred^2)
    d_bearing_dy <- x_pos / (range_pred^2)
    H_jacobian <- matrix(c(d_range_dx, d_range_dy, 0, 0,
                           d_bearing_dx, d_bearing_dy, 0, 0), nrow = 2, byrow = TRUE)
  }
  
  ## Innovation
  y_innov <- z[, k + 1] - z_pred
  # Normalize bearing difference to be within [-pi, pi]
  y_innov[2] <- ((y_innov[2] + pi) %% (2 * pi)) - pi
  
  ## Innovation covariance and Kalman Gain
  S <- H_jacobian %*% P_pred %*% t(H_jacobian) + R_meas
  K <- P_pred %*% t(H_jacobian) %*% solve(S)
  
  ## Update
  x_est[, k + 1] <- x_pred + K %*% y_innov
  P <- (diag(4) - K %*% H_jacobian) %*% P_pred
}

## Plotting Results
par(mfrow = c(2, 1))
plot(t, x_true[1, ], type = "l", col = "blue", lwd = 2,
     xlab = "Time (s)", ylab = "x position (m)", main = "Cart Position Tracking")
lines(t, x_est[1, ], col = "red", lwd = 2, lty = 2)
legend("topleft", legend = c("True", "Estimated"), col = c("blue", "red"), lty = c(1,2))

plot(t, x_true[2, ], type = "l", col = "blue", lwd = 2,
     xlab = "Time (s)", ylab = "y position (m)", main = "Pole Position Tracking")
lines(t, x_est[2, ], col = "red", lwd = 2, lty = 2)
legend("topleft", legend = c("True", "Estimated"), col = c("blue", "red"), lty = c(1,2))

# Additional plots for velocities
par(mfrow = c(2, 1))
plot(t, x_true[3, ], type = "l", col = "blue", lwd = 2,
     xlab = "Time (s)", ylab = "x velocity (m/s)", main = "Cart Velocity Tracking")
lines(t, x_est[3, ], col = "red", lwd = 2, lty = 2)
legend("topleft", legend = c("True", "Estimated"), col = c("blue", "red"), lty = c(1,2))

plot(t, x_true[4, ], type = "l", col = "blue", lwd = 2,
     xlab = "Time (s)", ylab = "y velocity (m/s)", main = "Pole Velocity Tracking")
lines(t, x_est[4, ], col = "red", lwd = 2, lty = 2)
legend("topleft", legend = c("True", "Estimated"), col = c("blue", "red"), lty = c(1,2))

# 2D Trajectory Plot
plot(x_true[1, ], x_true[2, ], type = "l", col = "blue", lwd = 2,
     xlab = "x position (m)", ylab = "y position (m)", main = "2D Trajectory Tracking")
lines(x_est[1, ], x_est[2, ], col = "red", lwd = 2, lty = 2)
legend("topright", legend = c("True trajectory", "Estimated trajectory"), col = c("blue", "red"), lty = c(1,2))
grid()
