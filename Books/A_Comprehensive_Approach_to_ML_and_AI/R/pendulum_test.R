# Pendulum Test in R
#
# This script creates an instance of an Inverted Pendulum environment,
# resets it, and then steps through the episode by taking random actions.
# At each step, it displays the observation and reward, and renders the pendulum.
#
# Note: The InvertedPendulum environment is assumed to be defined as an R object
# with methods: reset(), step(action), render(), and a property maxTorque.

# For demonstration, we assume an InvertedPendulum environment exists.
# (You need to implement or load your own InvertedPendulum class in R.)

# Example stub (uncomment and adapt if needed):
# InvertedPendulum <- function() {
#   env <- list(
#     maxTorque = 2,
#     dt = 0.05,
#     state = c(0, 0),  # e.g., [theta, theta_dot]
#     reset = function() { env$state <<- c(0, 0); return(env$state) },
#     step = function(action) {
#       # Update env$state based on action (this is a stub)
#       env$state <<- env$state + action * env$dt  # not physically accurate
#       reward <- -abs(env$state[1])
#       done <- abs(env$state[1]) > pi/2
#       list(obs = env$state, reward = reward, done = done)
#     },
#     render = function() { print(paste("State:", paste(round(env$state, 3), collapse = ", "))) }
#   )
#   return(env)
# }

# Assuming the environment is defined, we instantiate it:
env <- InvertedPendulum()  # Replace with your environment constructor

# Reset environment
obs <- env$reset()
done <- FALSE

while (!done) {
  # Choose a random action between -maxTorque and maxTorque
  action <- -env$maxTorque + 2 * env$maxTorque * runif(1)
  # Take a step (assume env$step returns a list with elements obs, reward, done)
  stepResult <- env$step(action)
  obs <- stepResult$obs
  reward <- stepResult$reward
  done <- stepResult$done
  cat("Observation:", paste(round(obs, 3), collapse = ", "), ", Reward:", reward, "\n")
  env$render()
  # Optionally pause for visualization
  # Sys.sleep(0.01)
}

cat("Episode finished.\n")
