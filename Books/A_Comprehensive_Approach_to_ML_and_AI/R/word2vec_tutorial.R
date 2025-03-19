# word2vec_tutorial.R
# Word2Vec Skip-Gram Model Tutorial in R
# This script implements a basic skip-gram model. It:
#   1. Preprocesses a small corpus.
#   2. Builds a vocabulary and maps words to indices.
#   3. Generates training pairs using a context window.
#   4. Trains word embeddings using SGD.
#   5. Projects embeddings to 2D via PCA and plots them.

set.seed(1)

# Step 1: Define and Preprocess Corpus
corpus <- c(
  "The quick brown fox jumps over the lazy dog",
  "I love natural language processing",
  "Word embeddings capture semantic similarity",
  "Deep learning for NLP is fascinating"
)
# Preprocess: lowercase and remove punctuation
corpus <- sapply(corpus, function(doc) {
  doc <- tolower(doc)
  gsub("[^a-z\\s]", "", doc)
})

# Split into words
words <- unlist(strsplit(corpus, "\\s+"))
words <- words[words != ""]

# Step 2: Build Vocabulary
vocab <- unique(words)
V <- length(vocab)
cat(sprintf("Vocabulary size: %d\n", V))
# Create mapping: word -> index
word2idx <- setNames(1:V, vocab)

# Step 3: Generate Training Pairs (Skip-Gram)
windowSize <- 2  # context window size
trainingPairs <- matrix(ncol = 2, nrow = 0)
for (i in 1:length(words)) {
  center <- word2idx[[words[i]]]
  for (j in max(1, i-windowSize):min(length(words), i+windowSize)) {
    if (j == i) next
    context <- word2idx[[words[j]]]
    trainingPairs <- rbind(trainingPairs, c(center, context))
  }
}
numPairs <- nrow(trainingPairs)
cat(sprintf("Number of training pairs: %d\n", numPairs))

# Step 4: Initialize Weight Matrices
D <- 10  # Embedding dimension
W_in <- 0.01 * matrix(rnorm(V * D), nrow = V)   # Input embeddings (V x D)
W_out <- 0.01 * matrix(rnorm(D * V), nrow = D)   # Output weights (D x V)

# Step 5: Train Skip-Gram Model using SGD
learningRate <- 0.05
numEpochs <- 1000
lossHistory <- numeric(numEpochs)

softmax <- function(x) {
  x <- x - max(x)
  exp_x <- exp(x)
  exp_x / sum(exp_x)
}

for (epoch in 1:numEpochs) {
  totalLoss <- 0
  idx <- sample(1:numPairs)  # shuffle pairs
  for (i in idx) {
    pair <- trainingPairs[i, ]
    center <- pair[1]
    context <- pair[2]
    
    # Forward pass: get embedding for center word
    h <- W_in[center, ]
    scores <- W_out %*% h  # scores for all words (D x 1 multiplied by D? Actually, W_out is D x V, so scores is V x 1)
    scores <- as.vector(scores)
    probs <- softmax(scores)
    
    # Loss: negative log likelihood of context word
    loss <- -log(probs[context] + 1e-10)
    totalLoss <- totalLoss + loss
    
    # Backpropagation
    dscores <- probs
    dscores[context] <- dscores[context] - 1
    # Gradient for W_out: outer product of h and dscores
    dW_out <- h %o% dscores  # (D x V)
    dh <- W_out %*% dscores   # (D x 1)
    
    # Update parameters
    W_in[center, ] <- W_in[center, ] - learningRate * as.vector(dh)
    W_out <- W_out - learningRate * dW_out
  }
  lossHistory[epoch] <- totalLoss / numPairs
  if (epoch %% 100 == 0) {
    cat(sprintf("Epoch %d/%d, Loss: %.4f\n", epoch, numEpochs, lossHistory[epoch]))
  }
}

# Step 6: Visualize Learned Word Embeddings with PCA
pca_result <- prcomp(W_in, scale. = FALSE)
plot(pca_result$x[,1:2], pch = 16, col = "blue", main = "Word Embeddings (2D PCA Projection)",
     xlab = "PC1", ylab = "PC2")
text(pca_result$x[,1], pca_result$x[,2], labels = names(word2idx), pos = 3)
