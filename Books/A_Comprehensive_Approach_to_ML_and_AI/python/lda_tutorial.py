import numpy as np
import matplotlib.pyplot as plt

def myLDA(docs, V, K, alpha, beta, T):
    """
    Perform Latent Dirichlet Allocation using collapsed Gibbs sampling.
    
    Parameters:
      docs  : list of arrays/lists of word indices for each document.
      V     : Vocabulary size.
      K     : Number of topics.
      alpha : Hyperparameter for document-topic distribution.
      beta  : Hyperparameter for topic-word distribution.
      T     : Number of Gibbs sampling iterations.
      
    Returns:
      theta : Document-topic distribution (M x K matrix).
      phi   : Topic-word distribution (K x V matrix).
      Z     : List of topic assignments for each document.
    """
    M = len(docs)
    ndk = np.zeros((M, K))    # document-topic counts
    nkw = np.zeros((K, V))    # topic-word counts
    nk = np.zeros(K)          # topic counts
    Z = []                    # topic assignments per document

    # Randomly initialize topic assignments for each word in each document
    for d, doc in enumerate(docs):
        N_d = len(doc)
        z_d = np.zeros(N_d, dtype=int)
        for n, w in enumerate(doc):
            topic = np.random.randint(K)
            z_d[n] = topic
            ndk[d, topic] += 1
            nkw[topic, w-1] += 1  # adjusting for Python 0-indexing (assumes words 1...V)
            nk[topic] += 1
        Z.append(z_d)

    # Gibbs sampling iterations
    for t in range(T):
        for d, doc in enumerate(docs):
            N_d = len(doc)
            for n, w in enumerate(doc):
                w_index = w - 1  # adjust word index for 0-indexing
                current_topic = Z[d][n]
                # Remove current assignment
                ndk[d, current_topic] -= 1
                nkw[current_topic, w_index] -= 1
                nk[current_topic] -= 1
                
                # Compute conditional probability for each topic
                p = (ndk[d, :] + alpha) * (nkw[:, w_index] + beta) / (nk + V * beta)
                p = p / p.sum()
                
                # Sample a new topic from multinomial distribution
                new_topic = np.argmax(np.random.multinomial(1, p))
                
                # Update counts and topic assignment
                Z[d][n] = new_topic
                ndk[d, new_topic] += 1
                nkw[new_topic, w_index] += 1
                nk[new_topic] += 1

    # Estimate theta (document-topic distribution)
    theta = (ndk + alpha)
    theta = theta / theta.sum(axis=1, keepdims=True)
    
    # Estimate phi (topic-word distribution)
    phi = (nkw + beta)
    phi = phi / phi.sum(axis=1, keepdims=True)
    
    return theta, phi, Z

# Example usage:
# Define a simple corpus: each document is a list of word indices (values between 1 and 15)
docs = [
    [1, 5, 3, 2, 7, 5],
    [4, 2, 5, 6, 8, 2, 5],
    [7, 8, 2, 3, 1, 4],
    [3, 6, 1, 7, 4, 2, 9]
]
V = 15
K = 4
alpha = 0.1
beta = 0.01
T = 100

theta, phi, Z = myLDA(docs, V, K, alpha, beta, T)

# Plot the topic distribution for the first document
plt.figure()
plt.bar(np.arange(1, K+1), theta[0, :])
plt.xlabel("Topic")
plt.ylabel("Probability")
plt.title("Topic Distribution for Document 1")
plt.show()
