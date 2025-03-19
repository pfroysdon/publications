# decision_tree_graph_tutorial.R
# This script builds a decision tree from data read from "data/data.txt" and plots a simple tree graph.
# (Note: This is a basic textual/graphical representation similar to the MATLAB version.)

rm(list=ls())
graphics.off()

# Read data from file (assumed tab-delimited, no header)
M <- as.matrix(read.table("data/data.txt", sep = "\t", header = FALSE))
Y <- M[, 1]
X <- M[, -1]
cols <- c("cyl", "dis", "hp", "wgt", "acc", "mtn", "mkr")

# Build the decision tree using recursive splitting
build_tree <- function(X, Y, cols) {
  inds <- list(1:nrow(X))
  p <- c(0)  # Parent indices; 0 for root
  labels <- c()
  result <- split_node(X, Y, inds, p, labels, cols, 1)
  list(inds = result$inds, p = result$p, labels = result$labels)
}

split_node <- function(X, Y, inds, p, labels, cols, node) {
  if (length(unique(Y[inds[[node]]])) == 1) return(list(inds = inds, p = p, labels = labels))
  if (nrow(unique(X[inds[[node]], , drop = FALSE])) == 1) return(list(inds = inds, p = p, labels = labels))
  
  best_ig <- -Inf
  best_feature <- NA
  best_val <- NA
  
  curr_X <- X[inds[[node]], , drop = FALSE]
  curr_Y <- Y[inds[[node]]]
  
  for (i in 1:ncol(X)) {
    feat <- curr_X[, i]
    vals <- sort(unique(feat))
    if (length(vals) < 2) next
    splits <- 0.5 * (vals[-length(vals)] + vals[-1])
    for (thresh in splits) {
      bin_vec <- feat < thresh
      ig <- entropy(curr_Y) - cond_ent(curr_Y, bin_vec)
      if (ig > best_ig) {
        best_ig <- ig
        best_feature <- i
        best_val <- thresh
        best_split <- bin_vec
      }
    }
  }
  
  if (is.na(best_feature)) return(list(inds = inds, p = p, labels = labels))
  
  # Split node: assign left and right indices
  left_inds <- inds[[node]][best_split]
  right_inds <- inds[[node]][!best_split]
  inds[[node]] <- NULL
  inds <- c(inds, list(left_inds), list(right_inds))
  p <- c(p, node, node)
  labels <- c(labels, sprintf("%s < %.2f", cols[best_feature], best_val),
              sprintf("%s >= %.2f", cols[best_feature], best_val))
  
  n <- length(p)
  result1 <- split_node(X, Y, inds, p, labels, cols, n - 1)
  result2 <- split_node(X, Y, result1$inds, result1$p, result1$labels, cols, n)
  result2
}

entropy <- function(Y) {
  tab <- table(Y)
  prob <- tab / sum(tab)
  -sum(prob * log2(prob + 1e-10))
}

cond_ent <- function(Y, X_bin) {
  result <- 0
  for (val in unique(X_bin)) {
    indices <- which(X_bin == val)
    H <- entropy(Y[indices])
    prob <- length(indices) / length(Y)
    result <- result + prob * H
  }
  result
}

# Build tree
t <- build_tree(X, Y, cols)

# Plot a simple representation of the tree (text labels)
plot.new()
plot.window(xlim = c(0, 1), ylim = c(0, 1))
title("Decision Tree (\"**\" indicates inconsistent node)")
text(0.5, 0.9, paste("Root: indices", paste(t$inds[[1]], collapse = ",")))
if (length(t$p) > 1) {
  for (i in 2:length(t$p)) {
    text(runif(1), runif(1), t$labels[i - 1])
  }
}
