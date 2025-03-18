import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

# Parameters.
seq_len = 8         # number of tokens
d_model = 16        # model (embedding) dimension
num_heads = 4       # number of attention heads (must divide d_model)
d_k = int(d_model / num_heads)  # dimension per head

# Generate random Q, K, V matrices: shape (seq_len, d_model)
Q = np.random.randn(seq_len, d_model)
K = np.random.randn(seq_len, d_model)
V = np.random.randn(seq_len, d_model)

def scaled_dot_product_attention(Q, K, V):
    """Compute scaled dot-product attention."""
    d_k = Q.shape[1]
    scores = Q @ K.T / np.sqrt(d_k)
    # Apply softmax row-wise.
    exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    attn_weights = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    output = attn_weights @ V
    return output, attn_weights

# Compute attention for each head.
head_weights = []
for h in range(num_heads):
    idx_start = h * d_k
    idx_end = (h+1) * d_k
    Q_h = Q[:, idx_start:idx_end]
    K_h = K[:, idx_start:idx_end]
    V_h = V[:, idx_start:idx_end]
    output, attn = scaled_dot_product_attention(Q_h, K_h, V_h)
    head_weights.append(attn)

# Plot attention weights for each head.
plt.figure(figsize=(8,8))
for h in range(num_heads):
    plt.subplot(2,2,h+1)
    plt.imshow(head_weights[h], cmap='viridis')
    plt.colorbar()
    plt.title(f'Head {h+1} Attention Weights')
    plt.xlabel('Key Index')
    plt.ylabel('Query Index')
plt.suptitle('Multi-Head Attention Weights Visualization')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
