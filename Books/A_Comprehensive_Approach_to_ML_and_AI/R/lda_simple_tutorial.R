# Clear workspace and close figures
rm(list = ls())
graphics.off()

# ------------------------------
# Define a Simple Corpus (Documents represented by word indices)
# ------------------------------
docs <- list(c(1, 5, 3, 2, 7, 5),
             c(4, 2, 5, 6, 8, 2, 5),
             c(7, 8, 2, 3, 1, 4),
             c(3, 6, 1, 7, 4, 2, 9))

# ------------------------------
# Define LDA Function using Collapsed Gibbs Sampling
# ------------------------------
myLDA <- function(docs, V, K, alpha, beta, T) {
  M <- length(docs)
  ndk <- matrix(0, nrow = M, ncol = K)  # document-topic counts
  nkw <- matrix(0, nrow = K, ncol = V)  # topic-word counts
  nk <- rep(0, K)                      # topic counts
  
  Z <- vector("list", M)  # topic assignments for each document
  
  # Randomly initialize topic assignments for each word in each document
  for (d in 1:M) {
    N_d <- length(docs[[d]])
    Z[[d]] <- rep(0, N_d)
    for (n in 1:N_d) {
      w <- docs[[d]][n]
      topic <- sample(1:K, 1)
      Z[[d]][n] <- topic
      ndk[d, topic] <- ndk[d, topic] + 1
      nkw[topic, w] <- nkw[topic, w] + 1
      nk[topic] <- nk[topic] + 1
    }
  }
  
  # Gibbs sampling iterations
  for (t in 1:T) {
    for (d in 1:M) {
      N_d <- length(docs[[d]])
      for (n in 1:N_d) {
        w <- docs[[d]][n]
        topic <- Z[[d]][n]
        
        # Remove current assignment
        ndk[d, topic] <- ndk[d, topic] - 1
        nkw[topic, w] <- nkw[topic, w] - 1
        nk[topic] <- nk[topic] - 1
        
        # Compute conditional distribution for each topic
        p <- numeric(K)
        for (k in 1:K) {
          p[k] <- (ndk[d, k] + alpha) * (nkw[k, w] + beta) / (nk[k] + V * beta)
        }
        p <- p / sum(p)
        
        # Sample new topic using the computed probabilities
        new_topic <- sample(1:K, 1, prob = p)
        
        # Update assignment and counts
        Z[[d]][n] <- new_topic
        ndk[d, new_topic] <- ndk[d, new_topic] + 1
        nkw[new_topic, w] <- nkw[new_topic, w] + 1
        nk[new_topic] <- nk[new_topic] + 1
      }
    }
  }
  
  # Estimate document-topic distributions (theta)
  theta <- ndk + alpha
  theta <- theta / rowSums(theta)
  
  # Estimate topic-word distributions (phi)
  phi <- nkw + beta
  phi <- phi / rowSums(phi)
  
  return(list(theta = theta, phi = phi, Z = Z))
}

# Run LDA with parameters: Vocabulary size = 15, Topics = 4, alpha = 0.1, beta = 0.01, iterations = 100
lda_result <- myLDA(docs, V = 15, K = 4, alpha = 0.1, beta = 0.01, T = 100)

# ------------------------------
# Plot the Topic Distribution for Document 1
# ------------------------------
barplot(lda_result$theta[1, ], main = "Topic Distribution for Document 1",
        xlab = "Topic", ylab = "Probability")
