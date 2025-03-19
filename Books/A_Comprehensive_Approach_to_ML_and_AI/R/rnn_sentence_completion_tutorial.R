# RNN Sentence Completion Tutorial in R using Keras
#
# Steps:
#   1. Load and preprocess "alice-in-wonderland.txt".
#   2. Tokenize text and build vocabulary.
#   3. Create training examples: for every three words, predict the fourth.
#   4. Split into training and validation sets.
#   5. Define and train an LSTM network for next-word prediction.
#   6. Inference: given a three-word prompt, predict the next word.

library(keras)
library(readr)

# 1. Load and Preprocess Text
text <- tolower(read_file("data/alice-in-wonderland.txt"))
text <- gsub("[^a-z0-9\\s]", "", text)
tokens <- unlist(strsplit(text, "\\s+"))
tokens <- tokens[tokens != ""]

# 2. Build Vocabulary and Mappings
vocab <- unique(tokens)
vocabSize <- length(vocab)
word2idx <- setNames(1:vocabSize, vocab)
idx2word <- setNames(vocab, 1:vocabSize)

# 3. Create Training Sequences (3 words as input, 4th as target)
sequenceLength <- 3
numSequences <- length(tokens) - sequenceLength
X <- list()
Y <- c()
for (i in 1:numSequences) {
  seq <- tokens[i:(i+sequenceLength-1)]
  X[[i]] <- sapply(seq, function(w) word2idx[[w]])
  Y[i] <- word2idx[[ tokens[i+sequenceLength] ]]
}

# 4. Split Data into Training and Validation Sets
set.seed(1)
numExamples <- length(X)
trainRatio <- 0.9
trainIdx <- sample(1:numExamples, size = round(trainRatio * numExamples))
valIdx <- setdiff(1:numExamples, trainIdx)
XTrain <- X[trainIdx]
YTrain <- Y[trainIdx]
XVal <- X[valIdx]
YVal <- Y[valIdx]

# 5. Define and Train LSTM Network using Keras
embeddingDim <- 50
numHiddenUnits <- 100

model <- keras_model_sequential() %>%
  layer_embedding(input_dim = vocabSize + 1, output_dim = embeddingDim, input_length = sequenceLength) %>%
  layer_lstm(units = numHiddenUnits) %>%
  layer_dense(units = vocabSize, activation = "softmax")
model %>% compile(loss = "sparse_categorical_crossentropy", optimizer = "adam", metrics = "accuracy")
model %>% summary()

history <- model %>% fit(
  x = do.call(rbind, XTrain),
  y = YTrain,
  batch_size = 128,
  epochs = 10,
  validation_data = list(do.call(rbind, XVal), YVal)
)

# 6. Inference: Given a three-word prompt, predict the next word.
completeSentence <- function(prompt, model, word2idx, idx2word) {
  prompt <- tolower(prompt)
  prompt <- gsub("[^a-z0-9\\s]", "", prompt)
  words <- unlist(strsplit(prompt, "\\s+"))
  if (length(words) != 3) stop("Prompt must contain exactly 3 words.")
  seq_idx <- sapply(words, function(w) word2idx[[w]])
  pred <- model %>% predict(matrix(seq_idx, nrow = 1))
  idx <- which.max(pred)
  idx2word[[as.character(idx)]]
}

cat("Example Predictions:\n")
prompt1 <- "alice was beginning"
next1 <- completeSentence(prompt1, model, word2idx, idx2word)
cat(sprintf("Prompt: \"%s\" --> Next word: %s\n", prompt1, next1))
prompt2 <- "today is very"
next2 <- completeSentence(prompt2, model, word2idx, idx2word)
cat(sprintf("Prompt: \"%s\" --> Next word: %s\n", prompt2, next2))
