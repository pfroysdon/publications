import numpy as np
import re

np.random.seed(1)

# 1. Define a small corpus.
corpus = [
    'The cat sat on the mat.',
    'Dogs are loyal and friendly.',
    'The weather today is sunny and bright.',
    'Artificial intelligence is transforming the world.',
    'MATLAB is a powerful tool for engineering and data analysis.',
    'Machine learning techniques can solve complex problems.'
]
num_docs = len(corpus)

def build_vocabulary(corpus):
    words = []
    for doc in corpus:
        doc_clean = re.sub(r'[^\w\s]', '', doc.lower())
        words.extend(doc_clean.split())
    vocab = sorted(list(set(words)))
    return vocab

vocab = build_vocabulary(corpus)
print(f"Vocabulary ({len(vocab)} words):")
print(vocab)

def text_to_vector(text, vocab):
    text_clean = re.sub(r'[^\w\s]', '', text.lower())
    words = text_clean.split()
    vec = np.zeros(len(vocab))
    for i, w in enumerate(vocab):
        vec[i] = words.count(w)
    return vec

query = 'What are the benefits of machine learning?'
query_vec = text_to_vector(query, vocab)

def cosine_similarity(v1, v2):
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0
    return np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2))

def retrieve_documents(query, corpus, vocab, k):
    query_vec = text_to_vector(query, vocab)
    sims = np.zeros(len(corpus))
    for i, doc in enumerate(corpus):
        doc_vec = text_to_vector(doc, vocab)
        sims[i] = cosine_similarity(query_vec, doc_vec)
    sorted_idx = np.argsort(-sims)
    top_idx = sorted_idx[:k]
    top_sims = sims[top_idx]
    return top_idx, top_sims

k = 2
retrieved_idx, similarities = retrieve_documents(query, corpus, vocab, k)
print("Top retrieved document indices:")
print(retrieved_idx)
print("Similarity scores:")
print(similarities)
retrieved_docs = [corpus[i] for i in retrieved_idx]

def generate_response(query, retrieved_docs):
    response = f'Your query was: "{query}". '
    if not retrieved_docs:
        response += "No relevant documents were found."
    else:
        response += "I found the following relevant information:"
        for doc in retrieved_docs:
            response += f" [{doc}]"
    return response

response = generate_response(query, retrieved_docs)
print("\nGenerated Response:")
print(response)
