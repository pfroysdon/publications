import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

np.random.seed(1)

# Step 1: Define and preprocess the corpus.
corpus = [
    'The quick brown fox jumps over the lazy dog',
    'I love natural language processing',
    'Word embeddings capture semantic similarity',
    'Deep learning for NLP is fascinating'
]
corpus = [re.sub(r'[^a-z\s]', '', s.lower()) for s in corpus]
words = []
for sentence in corpus:
    words.extend(sentence.split())

# Step 2: Build vocabulary.
vocab = sorted(set(words))
V = len(vocab)
print(f"Vocabulary size: {V}")
word2idx = {w: i for i, w in enumerate(vocab)}

# Step 3: Generate training pairs (skip-gram).
window_size = 2
training_pairs = []
for i, word in enumerate(words):
    center = word2idx[word]
    for j in range(max(0, i-window_size), min(len(words), i+window_size+1)):
        if j == i:
            continue
        context = word2idx[words[j]]
        training_pairs.append((center, context))
num_pairs = len(training_pairs)
print(f"Number of training pairs: {num_pairs}")

# Step 4: Initialize weight matrices.
D = 10  # embedding dimension
W_in = 0.01 * np.random.randn(V, D)   # Input embeddings.
W_out = 0.01 * np.random.randn(D, V)    # Output weights.

def softmax_vec(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

# Step 5: Train the skip-gram model using SGD.
learning_rate = 0.05
num_epochs = 1000
loss_history = np.zeros(num_epochs)

for epoch in range(num_epochs):
    total_loss = 0
    np.random.shuffle(training_pairs)
    for center, context in training_pairs:
        h = W_in[center, :].reshape(-1,1)  # (D,1)
        scores = W_out.T @ h  # (V,1)
        probs = softmax_vec(scores.flatten())
        loss = -np.log(probs[context] + 1e-10)
        total_loss += loss
        
        dscores = probs.copy()
        dscores[context] -= 1  # gradient of softmax loss.
        dW_out = h @ dscores.reshape(1,-1)  # (D,V)
        dh = W_out @ dscores  # (D,)
        
        # Update parameters.
        W_in[center, :] -= learning_rate * dh
        W_out -= learning_rate * dW_out
    loss_history[epoch] = total_loss / num_pairs
    if (epoch+1) % 100 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss_history[epoch]:.4f}")

# Step 6: Visualize learned embeddings using PCA.
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(W_in)
plt.figure(figsize=(8,8))
plt.scatter(embeddings_2d[:,0], embeddings_2d[:,1], s=50)
for i, word in enumerate(vocab):
    plt.text(embeddings_2d[i,0]+0.05, embeddings_2d[i,1], word, fontsize=12)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Learned Word Embeddings (2D PCA Projection)")
plt.grid(True)
plt.show()
