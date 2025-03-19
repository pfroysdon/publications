# Model Distillation Tutorial in R
#
# This tutorial demonstrates how to perform model distillation on a simple
# 2D classification problem. First, a teacher network (2 -> 20 -> 2) is trained
# on a synthetic dataset with two classes. Then, a smaller student network (2 -> 5 -> 2)
# is trained to mimic the teacher’s behavior by minimizing a combined loss:
#
#    Loss = lambda * HardLoss + (1 - lambda) * T^2 * DistillLoss
#
# where HardLoss is the cross–entropy loss between the student’s predictions and the true labels,
# and DistillLoss is the KL divergence between the teacher’s and student’s softened probability distributions.
#
# Finally, decision boundaries for both teacher and student are visualized.

set.seed(1)

# -----------------------------
# 1. Generate Synthetic Dataset
# -----------------------------
N <- 200  # total number of samples
# Class 1: Gaussian centered at (1,1)
X1 <- matrix(rnorm(2 * (N/2), mean = 0, sd = 0.5), nrow = 2) + matrix(rep(c(1, 1), N/2), nrow = 2)
# Class 2: Gaussian centered at (3,3)
X2 <- matrix(rnorm(2 * (N/2), mean = 0, sd = 0.5), nrow = 2) + matrix(rep(c(3, 3), N/2), nrow = 2)
X <- cbind(X1, X2)  # 2 x N data matrix
labels <- c(rep(1, N/2), rep(2, N/2))  # labels: 1 or 2

# One-hot encoding (Y: 2 x N)
Y <- matrix(0, nrow = 2, ncol = N)
for (i in 1:N) {
  Y[labels[i], i] <- 1
}

# -----------------------------
# Define Activation Functions
# -----------------------------
softmax <- function(x) {
  # Compute softmax along each column
  exp_x <- exp(x - apply(x, 2, max))
  exp_x / matrix(colSums(exp_x), nrow = nrow(x), ncol = ncol(x), byrow = TRUE)
}

relu <- function(x) {
  pmax(x, 0)
}

reluDerivative <- function(x) {
  as.numeric(x > 0)
}

# -----------------------------
# 2. Train Teacher Network
# -----------------------------
# Teacher architecture: 2 -> 20 -> 2
inputDim <- 2
hiddenDim_teacher <- 20
outputDim <- 2

# Initialize teacher parameters
W1_teacher <- matrix(rnorm(hiddenDim_teacher * inputDim, sd = 0.01), nrow = hiddenDim_teacher)
b1_teacher <- matrix(0, nrow = hiddenDim_teacher, ncol = 1)
W2_teacher <- matrix(rnorm(outputDim * hiddenDim_teacher, sd = 0.01), nrow = outputDim)
b2_teacher <- matrix(0, nrow = outputDim, ncol = 1)

teacher_lr <- 0.01
numIter_teacher <- 5000

for (iter in 1:numIter_teacher) {
  # Forward pass
  Z1 <- W1_teacher %*% X + matrix(rep(b1_teacher, N), nrow = hiddenDim_teacher)
  H1 <- relu(Z1)
  logits_teacher <- W2_teacher %*% H1 + matrix(rep(b2_teacher, N), nrow = outputDim)
  probs_teacher <- softmax(logits_teacher)
  
  # Compute cross-entropy loss
  loss_teacher <- -mean(colSums(Y * log(probs_teacher + 1e-8)))
  
  # Backpropagation
  d_logits <- probs_teacher - Y
  grad_W2_teacher <- (d_logits %*% t(H1)) / N
  grad_b2_teacher <- matrix(rowMeans(d_logits), ncol = 1)
  
  d_H1 <- t(W2_teacher) %*% d_logits
  d_Z1 <- d_H1 * matrix(reluDerivative(Z1), nrow = hiddenDim_teacher)
  grad_W1_teacher <- (d_Z1 %*% t(X)) / N
  grad_b1_teacher <- matrix(rowMeans(d_Z1), ncol = 1)
  
  # Parameter updates
  W2_teacher <- W2_teacher - teacher_lr * grad_W2_teacher
  b2_teacher <- b2_teacher - teacher_lr * grad_b2_teacher
  W1_teacher <- W1_teacher - teacher_lr * grad_W1_teacher
  b1_teacher <- b1_teacher - teacher_lr * grad_b1_teacher
  
  if (iter %% 500 == 0) {
    cat(sprintf("Teacher Iteration %d, Loss: %.4f\n", iter, loss_teacher))
  }
}

# Freeze teacher parameters
Teacher <- list(W1 = W1_teacher, b1 = b1_teacher, W2 = W2_teacher, b2 = b2_teacher, T = 2)

# -----------------------------
# 3. Train Student Network with Distillation
# -----------------------------
# Student architecture: 2 -> 5 -> 2
hiddenDim_student <- 5

W1_student <- matrix(rnorm(hiddenDim_student * inputDim, sd = 0.01), nrow = hiddenDim_student)
b1_student <- matrix(0, nrow = hiddenDim_student, ncol = 1)
W2_student <- matrix(rnorm(outputDim * hiddenDim_student, sd = 0.01), nrow = outputDim)
b2_student <- matrix(0, nrow = outputDim, ncol = 1)

student_lr <- 0.01
numIter_student <- 5000
lambda <- 0.5  # weight for hard loss vs. distillation loss
T_temp <- Teacher$T

for (iter in 1:numIter_student) {
  # Teacher forward pass (with temperature)
  Z1_teacher <- Teacher$W1 %*% X + matrix(rep(Teacher$b1, N), nrow = hiddenDim_teacher)
  H1_teacher <- relu(Z1_teacher)
  logits_teacher <- Teacher$W2 %*% H1_teacher + matrix(rep(Teacher$b2, N), nrow = outputDim)
  logits_teacher_T <- logits_teacher / T_temp
  soft_targets <- softmax(logits_teacher_T)
  
  # Student forward pass (with temperature)
  Z1_student <- W1_student %*% X + matrix(rep(b1_student, N), nrow = hiddenDim_student)
  H1_student <- relu(Z1_student)
  logits_student <- W2_student %*% H1_student + matrix(rep(b2_student, N), nrow = outputDim)
  logits_student_T <- logits_student / T_temp
  student_soft <- softmax(logits_student_T)
  
  # Student hard predictions (without temperature)
  student_probs <- softmax(logits_student)
  
  # Compute hard loss (cross-entropy) and distillation loss (KL divergence)
  hard_loss <- -mean(colSums(Y * log(student_probs + 1e-8)))
  kl_div <- colSums(soft_targets * (log(soft_targets + 1e-8) - log(student_soft + 1e-8)))
  distill_loss <- mean(kl_div)
  
  # Combined loss with temperature scaling T^2
  loss_student <- lambda * hard_loss + (1 - lambda) * T_temp^2 * distill_loss
  
  # Backpropagation for student network
  d_logits_hard <- student_probs - Y
  d_logits_distill <- (student_soft - soft_targets) * (1 / T_temp)
  d_logits_student <- lambda * d_logits_hard + (1 - lambda) * T_temp^2 * d_logits_distill
  
  grad_W2_student <- (d_logits_student %*% t(H1_student)) / N
  grad_b2_student <- matrix(rowMeans(d_logits_student), ncol = 1)
  
  d_H1_student <- t(W2_student) %*% d_logits_student
  d_Z1_student <- d_H1_student * matrix(reluDerivative(Z1_student), nrow = hiddenDim_student)
  grad_W1_student <- (d_Z1_student %*% t(X)) / N
  grad_b1_student <- matrix(rowMeans(d_Z1_student), ncol = 1)
  
  # Update student parameters
  W2_student <- W2_student - student_lr * grad_W2_student
  b2_student <- b2_student - student_lr * grad_b2_student
  W1_student <- W1_student - student_lr * grad_W1_student
  b1_student <- b1_student - student_lr * grad_b1_student
  
  if (iter %% 500 == 0) {
    cat(sprintf("Student Iteration %d, Hard Loss: %.4f, Distill Loss: %.4f, Combined Loss: %.4f\n",
                iter, hard_loss, distill_loss, loss_student))
  }
}

# -----------------------------
# 4. Evaluate and Visualize Decision Boundaries
# -----------------------------
# Create a grid over the input space
xGrid <- seq(min(X[1, ]) - 1, max(X[1, ]) + 1, length.out = 100)
yGrid <- seq(min(X[2, ]) - 1, max(X[2, ]) + 1, length.out = 100)
gridPoints <- rbind(as.vector(outer(xGrid, rep(1,100))),
                    as.vector(outer(rep(1,100), yGrid)))

# Teacher predictions on grid
Z1_teacher_grid <- Teacher$W1 %*% gridPoints + matrix(rep(Teacher$b1, ncol(gridPoints)), nrow = hiddenDim_teacher)
H1_teacher_grid <- relu(Z1_teacher_grid)
logits_teacher_grid <- Teacher$W2 %*% H1_teacher_grid + matrix(rep(Teacher$b2, ncol(gridPoints)), nrow = outputDim)
probs_teacher_grid <- softmax(logits_teacher_grid)
pred_teacher <- apply(probs_teacher_grid, 2, which.max)

# Student predictions on grid
Z1_student_grid <- W1_student %*% gridPoints + matrix(rep(b1_student, ncol(gridPoints)), nrow = hiddenDim_student)
H1_student_grid <- relu(Z1_student_grid)
logits_student_grid <- W2_student %*% H1_student_grid + matrix(rep(b2_student, ncol(gridPoints)), nrow = outputDim)
probs_student_grid <- softmax(logits_student_grid)
pred_student <- apply(probs_student_grid, 2, which.max)

# Reshape predictions for plotting
pred_teacher_grid_mat <- matrix(pred_teacher, nrow = 100, ncol = 100)
pred_student_grid_mat <- matrix(pred_student, nrow = 100, ncol = 100)

# Plot teacher and student decision boundaries
par(mfrow = c(1,2))
image(xGrid, yGrid, pred_teacher_grid_mat, col = terrain.colors(2), main = "Teacher Decision Boundary",
      xlab = "Feature 1", ylab = "Feature 2")
image(xGrid, yGrid, pred_student_grid_mat, col = terrain.colors(2), main = "Student Decision Boundary",
      xlab = "Feature 1", ylab = "Feature 2")
