# transformer_tutorial.R
# This tutorial implements a simplified Transformer network from scratch.
# It defines positional encoding, scaled dot-product attention, multi-head attention,
# a feed-forward network, layer normalization, and constructs encoder and decoder blocks.
# Finally, it assembles a small Transformer and performs a forward pass.

set.seed(1)

# Hyperparameters
seq_len_enc <- 10   # Encoder sequence length
seq_len_dec <- 10   # Decoder sequence length
d_model <- 32       # Model dimension
num_heads <- 4      # Number of attention heads
d_ff <- 64          # Feed-forward inner dimension
num_enc_layers <- 2
num_dec_layers <- 2
epsilon <- 1e-6

# Dummy input sequences (encoder and decoder inputs)
encoder_input <- matrix(rnorm(seq_len_enc * d_model), nrow = seq_len_enc, ncol = d_model)
decoder_input <- matrix(rnorm(seq_len_dec * d_model), nrow = seq_len_dec, ncol = d_model)

# Positional Encoding function
positionalEncoding <- function(seq_len, d_model) {
  posEnc <- matrix(0, nrow = seq_len, ncol = d_model)
  for (pos in 1:seq_len) {
    for (i in 1:d_model) {
      angle <- pos / (10000^((i-1)/d_model))
      if (i %% 2 == 1) {
        posEnc[pos, i] <- sin(angle)
      } else {
        posEnc[pos, i] <- cos(angle)
      }
    }
  }
  posEnc
}

# Add positional encodings
encoder_input <- encoder_input + positionalEncoding(seq_len_enc, d_model)
decoder_input <- decoder_input + positionalEncoding(seq_len_dec, d_model)

# Scaled Dot-Product Attention function
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

# Multi-Head Attention function
multiHeadAttention <- function(Q, K, V, num_heads, Wq, Wk, Wv, Wo, mask = NULL) {
  seq_len <- nrow(Q)
  d_model <- ncol(Q)
  d_k <- d_model / num_heads
  
  # Linear projections
  Q_proj <- Q %*% Wq
  K_proj <- K %*% Wk
  V_proj <- V %*% Wv
  
  # Split into heads
  Q_heads <- array(t(Q_proj), dim = c(d_k, num_heads, seq_len))
  K_heads <- array(t(K_proj), dim = c(d_k, num_heads, seq_len))
  V_heads <- array(t(V_proj), dim = c(d_k, num_heads, seq_len))
  
  head_outputs <- matrix(0, nrow = seq_len, ncol = d_model)
  for (h in 1:num_heads) {
    Q_h <- t(Q_heads[, h, ])
    K_h <- t(K_heads[, h, ])
    V_h <- t(V_heads[, h, ])
    attn_res <- scaledDotProductAttention(Q_h, K_h, V_h, mask)
    head_outputs[, ((h-1)*d_k + 1):(h*d_k)] <- attn_res$output
  }
  # Final linear projection
  head_outputs %*% Wo
}

# Position-wise Feed-Forward Network
positionwiseFeedForward <- function(X, W1, b1, W2, b2) {
  hidden <- pmax(X %*% W1 + matrix(rep(b1, nrow(X)), nrow = nrow(X), byrow = TRUE), 0)
  hidden %*% W2 + matrix(rep(b2, nrow(X)), nrow = nrow(X), byrow = TRUE)
}

# Layer Normalization
layerNorm <- function(X, epsilon = 1e-6) {
  mu <- rowMeans(X)
  sigma <- apply(X, 1, sd)
  (X - mu) / (sigma + epsilon)
}

# Encoder Block
encoderBlock <- function(X, num_heads, d_model, d_ff, Wq, Wk, Wv, Wo, W1, b1, W2, b2, epsilon) {
  # Multi-head self-attention
  attn_out <- multiHeadAttention(X, X, X, num_heads, Wq, Wk, Wv, Wo)
  out1 <- layerNorm(X + attn_out, epsilon)
  # Feed-forward network
  ff_out <- positionwiseFeedForward(out1, W1, b1, W2, b2)
  layerNorm(out1 + ff_out, epsilon)
}

# Decoder Block
decoderBlock <- function(X, encoderOutput, num_heads, d_model, d_ff,
                         Wq_self, Wk_self, Wv_self, Wo_self,
                         Wq_encdec, Wk_encdec, Wv_encdec, Wo_encdec,
                         W1, b1, W2, b2, epsilon, mask) {
  # Masked self-attention
  self_attn <- multiHeadAttention(X, X, X, num_heads, Wq_self, Wk_self, Wv_self, Wo_self, mask)
  out1 <- layerNorm(X + self_attn, epsilon)
  # Encoder-decoder attention
  encdec_attn <- multiHeadAttention(out1, encoderOutput, encoderOutput, num_heads,
                                    Wq_encdec, Wk_encdec, Wv_encdec, Wo_encdec)
  out2 <- layerNorm(out1 + encdec_attn, epsilon)
  # Feed-forward network
  ff_out <- positionwiseFeedForward(out2, W1, b1, W2, b2)
  layerNorm(out2 + ff_out, epsilon)
}

# Softmax function (for matrices, applied row-wise)
softmax_matrix <- function(X, dim = 2) {
  apply(X, 1, function(x) {
    x <- x - max(x)
    exp_x <- exp(x)
    exp_x / sum(exp_x)
  }) -> S
  t(S)
}

# Create a look-ahead mask for the decoder (lower triangular mask)
mask <- matrix(-Inf, nrow = seq_len_dec, ncol = seq_len_dec)
mask[lower.tri(mask, diag = TRUE)] <- 0

# Initialize encoder layer weights (for each layer, here 2 layers)
enc_layers <- vector("list", num_enc_layers)
for (i in 1:num_enc_layers) {
  enc_layers[[i]] <- list(
    Wq = matrix(rnorm(d_model * d_model), nrow = d_model),
    Wk = matrix(rnorm(d_model * d_model), nrow = d_model),
    Wv = matrix(rnorm(d_model * d_model), nrow = d_model),
    Wo = matrix(rnorm(d_model * d_model), nrow = d_model),
    W1 = matrix(rnorm(d_model * d_ff), nrow = d_model),
    b1 = rnorm(d_ff),
    W2 = matrix(rnorm(d_ff * d_model), nrow = d_ff),
    b2 = rnorm(d_model)
  )
}

# Initialize decoder layer weights (for each layer, here 2 layers)
dec_layers <- vector("list", num_dec_layers)
for (i in 1:num_dec_layers) {
  dec_layers[[i]] <- list(
    Wq_self = matrix(rnorm(d_model * d_model), nrow = d_model),
    Wk_self = matrix(rnorm(d_model * d_model), nrow = d_model),
    Wv_self = matrix(rnorm(d_model * d_model), nrow = d_model),
    Wo_self = matrix(rnorm(d_model * d_model), nrow = d_model),
    Wq_encdec = matrix(rnorm(d_model * d_model), nrow = d_model),
    Wk_encdec = matrix(rnorm(d_model * d_model), nrow = d_model),
    Wv_encdec = matrix(rnorm(d_model * d_model), nrow = d_model),
    Wo_encdec = matrix(rnorm(d_model * d_model), nrow = d_model),
    W1 = matrix(rnorm(d_model * d_ff), nrow = d_model),
    b1 = rnorm(d_ff),
    W2 = matrix(rnorm(d_ff * d_model), nrow = d_ff),
    b2 = rnorm(d_model)
  )
}

# Encoder forward pass
enc_output <- encoder_input
for (i in 1:num_enc_layers) {
  layer <- enc_layers[[i]]
  enc_output <- encoderBlock(enc_output, num_heads, d_model, d_ff,
                             layer$Wq, layer$Wk, layer$Wv, layer$Wo,
                             layer$W1, layer$b1, layer$W2, layer$b2, epsilon)
}

# Decoder forward pass
dec_output <- decoder_input
for (i in 1:num_dec_layers) {
  layer <- dec_layers[[i]]
  dec_output <- decoderBlock(dec_output, enc_output, num_heads, d_model, d_ff,
                             layer$Wq_self, layer$Wk_self, layer$Wv_self, layer$Wo_self,
                             layer$Wq_encdec, layer$Wk_encdec, layer$Wv_encdec, layer$Wo_encdec,
                             layer$W1, layer$b1, layer$W2, layer$b2, epsilon, mask)
}

cat("Transformer Network Output (first 3 tokens of decoder):\n")
print(dec_output[1:3, ])
