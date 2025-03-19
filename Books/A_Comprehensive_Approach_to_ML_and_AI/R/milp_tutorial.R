# MILP Tutorial: Solving a Traveling Salesman Problem (TSP) using MILP in R
#
# This script formulates and solves a TSP using the Miller–Tucker–Zemlin (MTZ)
# formulation with the lpSolve package. The TSP is defined on a set of cities with
# (x, y) coordinates. The goal is to find the shortest route that visits each city
# exactly once and returns to the starting city.

if (!requireNamespace("lpSolve", quietly = TRUE)) install.packages("lpSolve")
library(lpSolve)

set.seed(1)
numCities <- 10
# Define city coordinates (10 cities in a 100x100 region)
cityCoords <- matrix(runif(numCities * 2, min = 0, max = 100), ncol = 2)
plot(cityCoords, pch = 16, col = "blue", xlab = "X Coordinate", ylab = "Y Coordinate",
     main = "City Locations")
text(cityCoords[, 1] + 1, cityCoords[, 2], labels = 1:numCities, col = "red", cex = 1.2)

# Compute Euclidean distance matrix
D <- as.matrix(dist(cityCoords))
diag(D) <- 0

# Formulate MILP using MTZ formulation
n <- numCities
numX <- n * n
numU <- n - 1
numVars <- numX + numU

# Indices for x and u variables
xInd <- 1:numX
uInd <- (numX + 1):numVars

# Objective: minimize total distance
f.obj <- c(as.vector(D), rep(0, numU))

# Binary variables: x are binary; u are continuous.
binary.vec <- rep(0, numVars)
binary.vec[xInd] <- 1  # x variables are binary

# Variable bounds
lb <- rep(0, numVars)
ub <- rep(1, numVars)
lb[uInd] <- 1; ub[uInd] <- n - 1

# Equality constraints: Outgoing and incoming edges & no self-loops
Aeq_out <- matrix(0, nrow = n, ncol = numVars)
for (i in 1:n) {
  idx <- ((i - 1) * n + 1):(i * n)
  Aeq_out[i, idx] <- 1
}
beq_out <- rep(1, n)

Aeq_in <- matrix(0, nrow = n, ncol = numVars)
for (j in 1:n) {
  idx <- seq(j, numX, by = n)
  Aeq_in[j, idx] <- 1
}
beq_in <- rep(1, n)

Aeq_self <- matrix(0, nrow = n, ncol = numVars)
for (i in 1:n) {
  Aeq_self[i, (i - 1) * n + i] <- 1
}
beq_self <- rep(0, n)

Aeq <- rbind(Aeq_out, Aeq_in, Aeq_self)
beq <- c(beq_out, beq_in, beq_self)

# Inequality constraints: MTZ subtour elimination constraints for i, j = 2,...,n (i ≠ j)
numMTZ <- (n - 1) * (n - 1)
Aineq <- matrix(0, nrow = numMTZ, ncol = numVars)
bineq <- rep(n - 1, numMTZ)
constraintCount <- 0
for (i in 2:n) {
  for (j in 2:n) {
    if (i != j) {
      constraintCount <- constraintCount + 1
      Aineq[constraintCount, numX + (i - 1)] <- 1      # u(i)
      Aineq[constraintCount, numX + (j - 1)] <- -1     # -u(j)
      Aineq[constraintCount, (i - 1) * n + j] <- n     # n * x(i,j)
    }
  }
}

# Combine constraints
const.mat <- rbind(Aineq, Aeq)
const.dir <- c(rep("<=", nrow(Aineq)), rep("=", nrow(Aeq)))
const.rhs <- c(bineq, beq)

# Solve the MILP
solution <- lp("min", f.obj, const.mat, const.dir, const.rhs,
               binary.vec = binary.vec, lower = lb, upper = ub, compute.sens = FALSE)
if (solution$status != 0) stop("The solver did not find an optimal solution.")

z <- solution$solution
# Extract x solution and reshape into matrix form
x_sol <- z[xInd]
X_matrix <- matrix(x_sol, nrow = n, byrow = TRUE)
X_matrix <- round(X_matrix)  # Enforce binary decisions

cat(sprintf("Optimal tour length: %.2f\n", solution$objval))
cat("Decision variable matrix X (rows: from city, columns: to city):\n")
print(X_matrix)

# Extract the tour from the decision matrix
tour <- numeric(n + 1)
currentCity <- 1
tour[1] <- currentCity
for (k in 2:n) {
  nextCity <- which(X_matrix[currentCity, ] == 1)
  tour[k] <- nextCity
  currentCity <- nextCity
}
tour[n + 1] <- tour[1]
cat("Optimal tour: ", paste(tour, collapse = " -> "), "\n")

# Visualization of the tour
plot(cityCoords, pch = 16, col = "black", xlab = "X Coordinate", ylab = "Y Coordinate",
     main = "Optimal TSP Tour")
text(cityCoords[, 1] + 1, cityCoords[, 2], labels = 1:n, col = "blue", cex = 1.2)
for (i in 1:n) {
  cityFrom <- tour[i]
  cityTo <- tour[i + 1]
  segments(cityCoords[cityFrom, 1], cityCoords[cityFrom, 2],
           cityCoords[cityTo, 1], cityCoords[cityTo, 2], col = "red", lwd = 2)
}
grid()
