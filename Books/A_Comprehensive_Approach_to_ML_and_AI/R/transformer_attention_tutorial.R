# transformer_attention_tutorial.R
# This tutorial visualizes the multi-head scaled dot-product attention weights.
# It generates random Q, K, V matrices, splits them into multiple heads, computes attention,
# and plots the weights for each head.

set.seed(1)

# Parameters
seq_len <- 8         # Number of tokens in the sequence
d_model <- 16        # Model (embedding) dimension
num_heads <- 4       # Number of attention heads
d_k <- d_model / num_heads  # Dimension per head

# Generate random Q, K, V matrices (seq_len x d_model)
Q <- matrix(rnorm(seq_len * d_model), nrow = seq_len)
K <- matrix(rnorm(seq_len * d_model), nrow = seq_len)
V <- matrix(rnorm(seq_len * d_model), nrow = seq_len)

# Scaled dot-product attention function
scaledDotProductAttention <- function(Q, K, V, mask = NULL) {
  d_k <- ncol(K)
  scores <- Q %*% t(K) / sqrt(d_k)
  if (!is.null(mask)) {
    scores <- scores + mask
  }
  attn_weights <- t(apply(scores, 1, function(x) {
    x <- x - max(x)
    exp_x <- exp(x)
    exp_x / sum(exp_x)
  }))
  output <- attn_weights %*% V
  list(output = output, attn = attn_weights)
}

# Multi-head attention: process each head separately
headWeights <- vector("list", num_heads)
for (h in 1:num_heads) {
  idx <- ((h-1)*d_k + 1):(h*d_k)
  Q_h <- Q[, idx]
  K_h <- K[, idx]
  V_h <- V[, idx]
  
  res <- scaledDotProductAttention(Q_h, K_h, V_h)
  headWeights[[h]] <- res$attn
}

# Plot attention weights for each head using image
par(mfrow = c(2,2))
for (h in 1:num_heads) {
  image(1:seq_len, 1:seq_len, headWeights[[h]], main = paste("Head", h, "Attention"),
        xlab = "Key Index", ylab = "Query Index", col = heat.colors(100))
  colorbar <- function() {
    z <- seq(min(headWeights[[h]]), max(headWeights[[h]]), length.out = 100)
    image(matrix(z, nrow=1), col=heat.colors(100), axes=FALSE)
  }
}
