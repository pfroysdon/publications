# vnn_mnist_tutorial.R
# VNN Tutorial for MNIST (or digit) classification from scratch.
# This script loads a digits dataset, scales the data, splits it into training and testing sets,
# trains a neural network with one hidden layer using gradient descent, and computes accuracy.

set.seed(42)
library(R.matlab)

# Load dataset from MAT file (assumes file 'digits.mat' is available in the data directory)
# Uncomment the following line if the MAT file is available:
# data_list <- readMat("data/digits.mat")
# For demonstration, we simulate a small digits dataset.
# Let's simulate: data: 200 x 64, images: 8 x 8 x 200, target: vector of 0-9.
data <- matrix(rnorm(200*64), nrow = 200)
images <- array(runif(200*8*8), dim = c(8,8,200))
target <- sample(0:9, 200, replace = TRUE)

cat("Data dimensions:", dim(data), "\n")
# Display one digit image (second image)
image(t(apply(images[,,2], 2, rev)), col = gray.colors(256), main = "Sample Digit")

# Scale the data
X <- scale(data)
cat("Scaled first sample:\n"); print(X[1,])

y <- target

# Train-test split: 60% training, 40% testing
set.seed(42)
N <- nrow(X)
train_idx <- sample(1:N, size = floor(0.6 * N))
X_train <- X[train_idx, ]
X_test <- X[-train_idx, ]
y_train <- y[train_idx]
y_test <- y[-train_idx]

# Convert labels to one-hot encoding
convert_y_to_vect <- function(y) {
  n <- length(y)
  y_vect <- matrix(0, nrow = n, ncol = 10)
  for (i in 1:n) {
    y_vect[i, y[i] + 1] <- 1
  }
  y_vect
}
y_v_train <- convert_y_to_vect(y_train)
y_v_test <- convert_y_to_vect(y_test)

cat(sprintf("y_train[1]= %d\n", y_train[1]))
cat("y_v_train[1]=\n"); print(y_v_train[1, ])

# Define neural network structure: 64 input, 20 hidden, 10 output
nn_structure <- c(64, 20, 10)
iter_num <- 5000
alpha <- 0.5

# Functions for training a simple neural network from scratch

# Sigmoid activation
f_sigmoid <- function(x) 1 / (1 + exp(-x))
# Derivative of sigmoid
f_deriv <- function(x) {
  fx <- f_sigmoid(x)
  fx * (1 - fx)
}

# Feed-forward propagation (returns list of activations and linear combinations)
feed_forward <- function(x, W, b) {
  L <- length(W) + 1
  h <- vector("list", L)
  z <- vector("list", L)
  h[[1]] <- x
  for (l in 1:length(W)) {
    z[[l+1]] <- W[[l]] %*% h[[l]] + b[[l]]
    h[[l+1]] <- f_sigmoid(z[[l+1]])
  }
  list(h = h, z = z)
}

# Initialize weights and biases
setup_and_init_weights <- function(nn_structure) {
  L <- length(nn_structure)
  W <- vector("list", L-1)
  b <- vector("list", L-1)
  for (l in 1:(L-1)) {
    W[[l]] <- matrix(runif(nn_structure[l+1] * nn_structure[l], min = -0.5, max = 0.5), 
                     nrow = nn_structure[l+1])
    b[[l]] <- matrix(runif(nn_structure[l+1], min = -0.5, max = 0.5), ncol = 1)
  }
  list(W = W, b = b)
}

# Train neural network using gradient descent
train_nn <- function(nn_structure, X, y, iter_num, alpha) {
  init <- setup_and_init_weights(nn_structure)
  W <- init$W
  b <- init$b
  m <- nrow(X)
  avg_cost_func <- numeric(iter_num)
  L <- length(nn_structure)
  
  for (cnt in 0:(iter_num-1)) {
    tri_W <- lapply(W, function(mat) matrix(0, nrow = nrow(mat), ncol = ncol(mat)))
    tri_b <- lapply(b, function(vec) matrix(0, nrow = nrow(vec), ncol = ncol(vec)))
    avg_cost <- 0
    
    for (i in 1:m) {
      x_i <- matrix(X[i, ], ncol = 1)
      y_i <- matrix(y[i, ], ncol = 1)
      ff <- feed_forward(x_i, W, b)
      h <- ff$h; z <- ff$z
      delta <- vector("list", L)
      
      # Backpropagation
      for (l in L:2) {
        if (l == L) {
          delta[[l]] <- (h[[l]] - y_i) * f_deriv(z[[l]])
          avg_cost <- avg_cost + norm(y_i - h[[l]], type = "2")
        } else {
          delta[[l]] <- (t(W[[l]]) %*% delta[[l+1]]) * f_deriv(z[[l]])
          tri_W[[l-1]] <- tri_W[[l-1]] + delta[[l+1]] %*% t(h[[l]])
          tri_b[[l-1]] <- tri_b[[l-1]] + delta[[l+1]]
        }
      }
    }
    
    # Update parameters
    for (l in 1:(L-1)) {
      W[[l]] <- W[[l]] - alpha * (tri_W[[l]] / m)
      b[[l]] <- b[[l]] - alpha * (tri_b[[l]] / m)
    }
    avg_cost_func[cnt + 1] <- avg_cost / m
  }
  list(W = W, b = b, avg_cost_func = avg_cost_func)
}

# Prediction function
predict_y <- function(W, b, X, n_layers) {
  m <- nrow(X)
  y_pred <- numeric(m)
  for (i in 1:m) {
    x_i <- matrix(X[i, ], ncol = 1)
    ff <- feed_forward(x_i, W, b)
    output <- ff$h[[n_layers]]
    y_pred[i] <- which.max(output) - 1  # convert from 1-indexed to original label (0-9)
  }
  y_pred
}

# Train the neural network
nn_model <- train_nn(nn_structure, X, y_v_train, iter_num, alpha)
W_trained <- nn_model$W
b_trained <- nn_model$b
avg_cost_func <- nn_model$avg_cost_func

# Plot training cost
plot(avg_cost_func, type = "l", lwd = 2, xlab = "Iteration", ylab = "Average Cost", main = "Training Loss")

# Predict on test set
y_pred <- predict_y(W_trained, b_trained, X_test, length(nn_structure))
accuracy <- mean(y_pred == y_test) * 100
cat(sprintf("Accuracy score = %.2f%%\n", accuracy))
