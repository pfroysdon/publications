import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

def positional_encoding(seq_len, d_model):
    pos_enc = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(d_model):
            angle = pos / (10000 ** ((i)/d_model))
            if i % 2 == 0:
                pos_enc[pos, i] = np.sin(angle)
            else:
                pos_enc[pos, i] = np.cos(angle)
    return pos_enc

def softmax(X, axis=1):
    X_max = np.max(X, axis=axis, keepdims=True)
    expX = np.exp(X - X_max)
    return expX / np.sum(expX, axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[1]
    scores = Q @ K.T / np.sqrt(d_k)
    if mask is not None:
        scores += mask  # assume mask already has -inf for masked positions
    attn_weights = softmax(scores, axis=1)
    output = attn_weights @ V
    return output, attn_weights

def multi_head_attention(Q, K, V, num_heads, Wq, Wk, Wv, Wo, mask=None):
    seq_len, d_model = Q.shape
    d_k = d_model // num_heads
    # Linear projections
    Q_proj = Q @ Wq  # (seq_len, d_model)
    K_proj = K @ Wk
    V_proj = V @ Wv
    # Split into heads
    Q_heads = Q_proj.reshape(seq_len, num_heads, d_k)
    K_heads = K_proj.reshape(seq_len, num_heads, d_k)
    V_heads = V_proj.reshape(seq_len, num_heads, d_k)
    
    head_outputs = []
    for h in range(num_heads):
        Q_h = Q_heads[:, h, :]  # (seq_len, d_k)
        K_h = K_heads[:, h, :]
        V_h = V_heads[:, h, :]
        head_out, _ = scaled_dot_product_attention(Q_h, K_h, V_h, mask)
        head_outputs.append(head_out)
    # Concatenate heads and project.
    concatenated = np.concatenate(head_outputs, axis=1)  # (seq_len, d_model)
    output = concatenated @ Wo
    return output

def positionwise_feed_forward(X, W1, b1, W2, b2):
    hidden = np.maximum(0, X @ W1 + b1)
    output = hidden @ W2 + b2
    return output

def layer_norm(X, epsilon=1e-6):
    mu = np.mean(X, axis=1, keepdims=True)
    sigma = np.std(X, axis=1, keepdims=True)
    return (X - mu) / (sigma + epsilon)

def encoder_block(X, num_heads, d_model, d_ff, Wq, Wk, Wv, Wo, W1, b1, W2, b2, epsilon):
    # Multi-head self-attention with residual connection and layer norm.
    attn_out = multi_head_attention(X, X, X, num_heads, Wq, Wk, Wv, Wo)
    out1 = layer_norm(X + attn_out, epsilon)
    # Feed-forward network with residual connection and layer norm.
    ff_out = positionwise_feed_forward(out1, W1, b1, W2, b2)
    output = layer_norm(out1 + ff_out, epsilon)
    return output

def decoder_block(X, enc_output, num_heads, d_model, d_ff,
                  Wq_self, Wk_self, Wv_self, Wo_self,
                  Wq_encdec, Wk_encdec, Wv_encdec, Wo_encdec,
                  W1, b1, W2, b2, epsilon, mask):
    # Masked self-attention.
    self_attn = multi_head_attention(X, X, X, num_heads, Wq_self, Wk_self, Wv_self, Wo_self, mask)
    out1 = layer_norm(X + self_attn, epsilon)
    # Encoder-decoder attention.
    encdec_attn = multi_head_attention(out1, enc_output, enc_output, num_heads,
                                        Wq_encdec, Wk_encdec, Wv_encdec, Wo_encdec)
    out2 = layer_norm(out1 + encdec_attn, epsilon)
    # Feed-forward network.
    ff_out = positionwise_feed_forward(out2, W1, b1, W2, b2)
    output = layer_norm(out2 + ff_out, epsilon)
    return output

# Hyperparameters
seq_len_enc = 10
seq_len_dec = 10
d_model = 32
num_heads = 4
d_ff = 64
num_enc_layers = 2
num_dec_layers = 2
epsilon_val = 1e-6

# Dummy input sequences.
encoder_input = np.random.randn(seq_len_enc, d_model)
decoder_input = np.random.randn(seq_len_dec, d_model)

# Add positional encodings.
encoder_input += positional_encoding(seq_len_enc, d_model)
decoder_input += positional_encoding(seq_len_dec, d_model)

# Initialize encoder layers.
enc_layers = []
for i in range(num_enc_layers):
    layer = {
        'Wq': np.random.randn(d_model, d_model),
        'Wk': np.random.randn(d_model, d_model),
        'Wv': np.random.randn(d_model, d_model),
        'Wo': np.random.randn(d_model, d_model),
        'W1': np.random.randn(d_model, d_ff),
        'b1': np.random.randn(1, d_ff),
        'W2': np.random.randn(d_ff, d_model),
        'b2': np.random.randn(1, d_model)
    }
    enc_layers.append(layer)

# Initialize decoder layers.
dec_layers = []
for i in range(num_dec_layers):
    layer = {
        'Wq_self': np.random.randn(d_model, d_model),
        'Wk_self': np.random.randn(d_model, d_model),
        'Wv_self': np.random.randn(d_model, d_model),
        'Wo_self': np.random.randn(d_model, d_model),
        'Wq_encdec': np.random.randn(d_model, d_model),
        'Wk_encdec': np.random.randn(d_model, d_model),
        'Wv_encdec': np.random.randn(d_model, d_model),
        'Wo_encdec': np.random.randn(d_model, d_model),
        'W1': np.random.randn(d_model, d_ff),
        'b1': np.random.randn(1, d_ff),
        'W2': np.random.randn(d_ff, d_model),
        'b2': np.random.randn(1, d_model)
    }
    dec_layers.append(layer)

# Encoder forward pass.
enc_output = encoder_input
for layer in enc_layers:
    enc_output = encoder_block(enc_output, num_heads, d_model, d_ff,
                               layer['Wq'], layer['Wk'], layer['Wv'], layer['Wo'],
                               layer['W1'], layer['b1'], layer['W2'], layer['b2'],
                               epsilon_val)

# Create lower-triangular mask for decoder.
mask = np.triu(np.ones((seq_len_dec, seq_len_dec)) * -np.inf, k=1)

# Decoder forward pass.
dec_output = decoder_input
for layer in dec_layers:
    dec_output = decoder_block(dec_output, enc_output, num_heads, d_model, d_ff,
                               layer['Wq_self'], layer['Wk_self'], layer['Wv_self'], layer['Wo_self'],
                               layer['Wq_encdec'], layer['Wk_encdec'], layer['Wv_encdec'], layer['Wo_encdec'],
                               layer['W1'], layer['b1'], layer['W2'], layer['b2'], epsilon_val, mask)

print("Transformer Network Output (first 3 tokens of decoder):")
print(dec_output[:3])
