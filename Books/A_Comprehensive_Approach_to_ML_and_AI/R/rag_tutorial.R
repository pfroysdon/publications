# Retrieval-Augmented Generation (RAG) Tutorial in R
# 
# Steps:
#   1. Build a small text corpus.
#   2. Construct a vocabulary.
#   3. Convert a query and each document to bag-of-words vectors.
#   4. Compute cosine similarity between the query vector and each document vector.
#   5. Retrieve the top-k most relevant documents.
#   6. Generate a response by combining the query and the retrieved documents.

set.seed(1)

# 1. Define a Small Corpus
corpus <- c(
  "The cat sat on the mat.",
  "Dogs are loyal and friendly.",
  "The weather today is sunny and bright.",
  "Artificial intelligence is transforming the world.",
  "MATLAB is a powerful tool for engineering and data analysis.",
  "Machine learning techniques can solve complex problems."
)
numDocs <- length(corpus)

# 2. Build Vocabulary from the Corpus
buildVocabulary <- function(corpus) {
  words <- unlist(lapply(corpus, function(doc) {
    # Convert to lowercase and remove punctuation
    doc <- tolower(doc)
    doc <- gsub("[^\\w\\s]", "", doc)
    unlist(strsplit(doc, "\\s+"))
  }))
  unique(words)
}
vocab <- buildVocabulary(corpus)
cat("Vocabulary (", length(vocab), " words):\n", paste(vocab, collapse = ", "), "\n\n")

# 3. Convert Text to Bag-of-Words Vector
textToVector <- function(text, vocab) {
  text <- tolower(text)
  text <- gsub("[^\\w\\s]", "", text)
  words <- unlist(strsplit(text, "\\s+"))
  vec <- sapply(vocab, function(w) sum(words == w))
  vec
}
query <- "What are the benefits of machine learning?"
queryVec <- textToVector(query, vocab)

# 4. Cosine Similarity Function
cosineSimilarity <- function(v1, v2) {
  if (sum(v1) == 0 || sum(v2) == 0) return(0)
  sum(v1 * v2) / (sqrt(sum(v1^2)) * sqrt(sum(v2^2)))
}

# 5. Retrieve Top-k Documents
retrieveDocuments <- function(query, corpus, vocab, k) {
  queryVec <- textToVector(query, vocab)
  sims <- sapply(corpus, function(doc) {
    docVec <- textToVector(doc, vocab)
    cosineSimilarity(queryVec, docVec)
  })
  sortedIdx <- order(sims, decreasing = TRUE)
  list(indices = sortedIdx[1:min(k, length(corpus))],
       similarities = sims[sortedIdx[1:min(k, length(corpus))]])
}
k <- 2
retrieval <- retrieveDocuments(query, corpus, vocab, k)
cat("Top", k, "retrieved document indices:\n")
print(retrieval$indices)
cat("Similarity scores:\n")
print(retrieval$similarities)
retrievedDocs <- corpus[retrieval$indices]

# 6. Generate a Response by Combining the Query and Retrieved Documents
generateResponse <- function(query, retrievedDocs) {
  response <- paste("Your query was:", query, "\nHere are some relevant documents:")
  for (doc in retrievedDocs) {
    response <- paste(response, "\n- ", doc)
  }
  response
}
response <- generateResponse(query, retrievedDocs)
cat("\nGenerated Response:\n", response, "\n")
